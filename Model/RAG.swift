//
//  RAG.swift
//  NoteRag
//
//  Created by Sanskar Thapa on 7/14/25.
//

import Foundation
import NaturalLanguage
import FoundationModels

class Document {
    let id: String
    let content: String
    var sentenceEmbeddings: [(sentence: String, embedding: [Double])]
    var chunks: [(text: String, embedding: [Double])]

    init(id: String, content: String) {
        self.id = id
        self.content = content
        self.sentenceEmbeddings = []
        self.chunks = []
    }
}

class RAG {
    private var documents: [Document] = []
    private let sentenceEmbedding: NLEmbedding
    private let languageRecognizer = NLLanguageRecognizer()
    
    // Configuration options
    private let chunkSize = 200 // characters per chunk
    private let chunkOverlap = 50 // overlap between chunks
    private let useChunking = true // Toggle between sentence and chunk-based retrieval
    
    init() {
        // Try to use sentence embedding first, fallback to word embedding
        if let sentenceModel = NLEmbedding.sentenceEmbedding(for: .english) {
            self.sentenceEmbedding = sentenceModel
        } else if let wordModel = NLEmbedding.wordEmbedding(for: .english) {
            self.sentenceEmbedding = wordModel
        } else {
            fatalError("Unable to load embedding model")
        }
    }
    
    func addDocument(_ document: Document) {
        if useChunking {
            // Chunk-based approach for longer documents
            let chunks = createChunks(from: document.content)
            var chunkEmbeddings: [(String, [Double])] = []
            
            for chunk in chunks {
                if let embedding = getEnhancedEmbedding(for: chunk) {
                    chunkEmbeddings.append((chunk, embedding))
                }
            }
            document.chunks = chunkEmbeddings
        } else {
            // Sentence-based approach
            let sentences = extractSentences(from: document.content)
            var sentenceEmbeddings: [(String, [Double])] = []
            
            for sentence in sentences {
                if let embedding = getEnhancedEmbedding(for: sentence) {
                    sentenceEmbeddings.append((sentence, embedding))
                }
            }
            document.sentenceEmbeddings = sentenceEmbeddings
        }
        
        documents.append(document)
    }
    
    func searchRelevantSentences(for query: String, limit: Int = 3) -> [(documentID: String, text: String, similarity: Double)] {
        guard let queryEmbedding = getEnhancedEmbedding(for: query) else { return [] }
        
        var scored: [(String, String, Double)] = []
        
        for doc in documents {
            if useChunking {
                for (chunk, embedding) in doc.chunks {
                    let sim = cosineSimilarity(queryEmbedding, embedding)
                    scored.append((doc.id, chunk, sim))
                }
            } else {
                for (sentence, embedding) in doc.sentenceEmbeddings {
                    let sim = cosineSimilarity(queryEmbedding, embedding)
                    scored.append((doc.id, sentence, sim))
                }
            }
        }
        
        // Apply reranking based on keyword overlap
        let rerankedScores = rerank(scored, query: query)
        let sorted = rerankedScores.sorted { $0.2 > $1.2 }
        
        return Array(sorted.prefix(limit))
    }
    
    func generateResponse(for query: String) async throws -> LanguageModelSession.Response<String> {
        let topResults = searchRelevantSentences(for: query)
        
        // Deduplicate and aggregate context from same documents
        var contextByDoc: [String: [String]] = [:]
        for result in topResults {
            contextByDoc[result.documentID, default: []].append(result.text)
        }
        
        let context = contextByDoc.map { docID, texts in
            "Document \(docID):\n" + texts.joined(separator: "\n")
        }.joined(separator: "\n\n")
        
        let prompt = """
        You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        \(context)
        
        Question: \(query)
        
        Instructions:
        1. Answer based solely on the information provided in the context
        2. If the context doesn't contain enough information, say so
        3. Be concise and accurate
        
        Answer:
        """
        
        print("Generated prompt:\n\(prompt)\n")
        
        let session = LanguageModelSession()
        let response = try await session.respond(to: prompt)
        
        return response
    }
    
    // MARK: - Enhanced Embedding Methods
    
    private func getEnhancedEmbedding(for text: String) -> [Double]? {
        let cleanedText = preprocessText(text)
        
        // Try sentence embedding first (if available)
        if let directEmbedding = sentenceEmbedding.vector(for: cleanedText),
           !directEmbedding.isEmpty {
            return normalizeVector(directEmbedding)
        }
        
        // Fallback to weighted word embeddings
        return getWeightedWordEmbedding(for: cleanedText)
    }
    
    private func getWeightedWordEmbedding(for text: String) -> [Double]? {
        let tagger = NLTagger(tagSchemes: [.lexicalClass, .lemma])
        tagger.string = text
        
        var weightedEmbeddings: [(embedding: [Double], weight: Double)] = []
        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace]
        
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lexicalClass, options: options) { tag, range in
            let word = String(text[range])
            let lemma = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lemma).0?.rawValue ?? word
            
            if let embedding = sentenceEmbedding.vector(for: lemma.lowercased()) {
                // Weight based on part of speech
                let weight = getWeight(for: tag)
                weightedEmbeddings.append((embedding, weight))
            }
            
            return true
        }
        
        guard !weightedEmbeddings.isEmpty else { return nil }
        
        // Calculate weighted average
        let dimension = weightedEmbeddings[0].embedding.count
        var result = Array(repeating: 0.0, count: dimension)
        var totalWeight = 0.0
        
        for (embedding, weight) in weightedEmbeddings {
            for i in 0..<dimension {
                result[i] += embedding[i] * weight
            }
            totalWeight += weight
        }
        
        // Normalize by total weight
        if totalWeight > 0 {
            result = result.map { $0 / totalWeight }
        }
        
        return normalizeVector(result)
    }
    
    private func getWeight(for tag: NLTag?) -> Double {
        switch tag {
        case .noun, .verb:
            return 2.0
        case .adjective, .adverb:
            return 1.5
        case .pronoun, .preposition, .conjunction:
            return 0.5
        default:
            return 1.0
        }
    }
    
    // MARK: - Text Processing Methods
    
    private func preprocessText(_ text: String) -> String {
        // Remove extra whitespace and normalize
        let normalized = text
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        
        return normalized
    }
    
    private func extractSentences(from text: String) -> [String] {
        var sentences: [String] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            return true
        }
        
        return sentences
    }
    
    private func createChunks(from text: String) -> [String] {
        var chunks: [String] = []
        let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        
        var currentChunk = ""
        var wordCount = 0
        
        for (index, word) in words.enumerated() {
            currentChunk += word + " "
            wordCount += 1
            
            // Check if we should create a new chunk
            if currentChunk.count >= chunkSize || index == words.count - 1 {
                chunks.append(currentChunk.trimmingCharacters(in: .whitespaces))
                
                // Create overlap for next chunk
                if index < words.count - 1 {
                    let overlapWords = max(0, wordCount - (chunkOverlap / 10)) // Approximate overlap
                    let startIndex = max(0, index - overlapWords + 1)
                    currentChunk = words[startIndex...index].joined(separator: " ") + " "
                    wordCount = overlapWords
                }
            }
        }
        
        return chunks
    }
    
    // MARK: - Reranking Methods
    
    private func rerank(_ results: [(String, String, Double)], query: String) -> [(String, String, Double)] {
        let queryWords = Set(query.lowercased().components(separatedBy: .whitespacesAndNewlines))
        
        return results.map { (docId, text, similarity) in
            let textWords = Set(text.lowercased().components(separatedBy: .whitespacesAndNewlines))
            let overlap = Double(queryWords.intersection(textWords).count) / Double(queryWords.count)
            
            // Combine cosine similarity with keyword overlap
            let combinedScore = (similarity * 0.7) + (overlap * 0.3)
            
            return (docId, text, combinedScore)
        }
    }
    
    // MARK: - Utility Methods
    
    private func normalizeVector(_ vector: [Double]) -> [Double] {
        let magnitude = sqrt(vector.map { $0 * $0 }.reduce(0, +))
        guard magnitude > 0 else { return vector }
        return vector.map { $0 / magnitude }
    }
    
    private func cosineSimilarity(_ v1: [Double], _ v2: [Double]) -> Double {
        guard v1.count == v2.count else { return 0 }
        let dotProduct = zip(v1, v2).map(*).reduce(0, +)
        let magnitude1 = sqrt(v1.map { $0 * $0 }.reduce(0, +))
        let magnitude2 = sqrt(v2.map { $0 * $0 }.reduce(0, +))
        guard magnitude1 > 0 && magnitude2 > 0 else { return 0 }
        return dotProduct / (magnitude1 * magnitude2)
    }
}
