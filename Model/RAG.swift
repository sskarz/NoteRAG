//
//  RAG.swift
//  NoteRag
//
//  Created by Sanskar Thapa on 7/14/25.
//

import Foundation
import NaturalLanguage
import FoundationModels
import CryptoKit

// MARK: - Cache Manager
class EmbeddingCache {
    private let cacheDirectory: URL
    private let cacheQueue = DispatchQueue(label: "com.noterag.embeddingcache", attributes: .concurrent)
    private var memoryCache = NSCache<NSString, NSData>()
    
    init() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.cacheDirectory = documentsPath.appendingPathComponent("EmbeddingCache")
        
        // Create cache directory if it doesn't exist
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        
        // Configure memory cache
        memoryCache.countLimit = 1000 // Limit number of cached items
        memoryCache.totalCostLimit = 50 * 1024 * 1024 // 50MB limit
    }
    
    func getCachedEmbedding(for text: String) -> [Double]? {
        let key = cacheKey(for: text)
        
        // Check memory cache first
        if let cachedData = memoryCache.object(forKey: key as NSString) as Data? {
            return deserializeEmbedding(from: cachedData)
        }
        
        // Check disk cache
        let fileURL = cacheDirectory.appendingPathComponent(key)
        if let data = try? Data(contentsOf: fileURL) {
            // Add to memory cache for faster future access
            memoryCache.setObject(data as NSData, forKey: key as NSString, cost: data.count)
            return deserializeEmbedding(from: data)
        }
        
        return nil
    }
    
    func cacheEmbedding(_ embedding: [Double], for text: String) {
        let key = cacheKey(for: text)
        let data = serializeEmbedding(embedding)
        
        cacheQueue.async(flags: .barrier) {
            // Save to memory cache
            self.memoryCache.setObject(data as NSData, forKey: key as NSString, cost: data.count)
            
            // Save to disk cache
            let fileURL = self.cacheDirectory.appendingPathComponent(key)
            try? data.write(to: fileURL)
        }
    }
    
    func clearCache() {
        memoryCache.removeAllObjects()
        try? FileManager.default.removeItem(at: cacheDirectory)
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
    }
    
    private func cacheKey(for text: String) -> String {
        let data = Data(text.utf8)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    private func serializeEmbedding(_ embedding: [Double]) -> Data {
        return try! JSONEncoder().encode(embedding)
    }
    
    private func deserializeEmbedding(from data: Data) -> [Double]? {
        return try? JSONDecoder().decode([Double].self, from: data)
    }
}

// MARK: - Document Class
class Document {
    let id: String
    let content: String
    var sentenceEmbeddings: [(sentence: String, embedding: [Double])]
    var chunks: [(text: String, embedding: [Double])]
    var isProcessed: Bool = false

    init(id: String, content: String) {
        self.id = id
        self.content = content
        self.sentenceEmbeddings = []
        self.chunks = []
    }
}

// MARK: - RAG Class
class RAG {
    private var documents: [Document] = []
    private let sentenceEmbedding: NLEmbedding
    private let languageRecognizer = NLLanguageRecognizer()
    private let embeddingCache = EmbeddingCache()
    private let processingQueue = DispatchQueue(label: "com.noterag.processing", attributes: .concurrent)
    private let documentQueue = DispatchQueue(label: "com.noterag.documents", attributes: .concurrent)
    
    // Configuration options
    private let chunkSize = 200 // characters per chunk
    private let chunkOverlap = 50 // overlap between chunks
    private let useChunking = true // Toggle between sentence and chunk-based retrieval
    private let maxConcurrentProcessing = 4 // Max documents to process simultaneously
    
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
    
    // MARK: - Async Document Processing
    
    func addDocument(_ document: Document) async {
        await withCheckedContinuation { continuation in
            addDocument(document) {
                continuation.resume()
            }
        }
    }
    
    func addDocument(_ document: Document, completion: @escaping () -> Void) {
        processingQueue.async {
            self.processDocument(document)
            
            self.documentQueue.async(flags: .barrier) {
                self.documents.append(document)
                completion()
            }
        }
    }
    
    func addDocuments(_ documents: [Document]) async {
        await withTaskGroup(of: Void.self) { group in
            // Limit concurrent processing
            let semaphore = DispatchSemaphore(value: maxConcurrentProcessing)
            
            for document in documents {
                group.addTask {
                    await withCheckedContinuation { continuation in
                        semaphore.wait()
                        self.addDocument(document) {
                            semaphore.signal()
                            continuation.resume()
                        }
                    }
                }
            }
        }
    }
    
    private func processDocument(_ document: Document) {
        if useChunking {
            // Chunk-based approach for longer documents
            let chunks = createChunks(from: document.content)
            var chunkEmbeddings: [(String, [Double])] = []
            
            // Process chunks in parallel
            let chunkGroup = DispatchGroup()
            let chunkQueue = DispatchQueue(label: "com.noterag.chunks", attributes: .concurrent)
            var tempEmbeddings = [(Int, String, [Double])]()
            
            for (index, chunk) in chunks.enumerated() {
                chunkGroup.enter()
                chunkQueue.async {
                    if let embedding = self.getCachedOrComputeEmbedding(for: chunk) {
                        chunkQueue.async(flags: .barrier) {
                            tempEmbeddings.append((index, chunk, embedding))
                        }
                    }
                    chunkGroup.leave()
                }
            }
            
            chunkGroup.wait()
            
            // Sort by original index to maintain order
            tempEmbeddings.sort { $0.0 < $1.0 }
            chunkEmbeddings = tempEmbeddings.map { ($0.1, $0.2) }
            
            document.chunks = chunkEmbeddings
        } else {
            // Sentence-based approach
            let sentences = extractSentences(from: document.content)
            var sentenceEmbeddings: [(String, [Double])] = []
            
            // Process sentences in parallel
            let sentenceGroup = DispatchGroup()
            let sentenceQueue = DispatchQueue(label: "com.noterag.sentences", attributes: .concurrent)
            var tempEmbeddings = [(Int, String, [Double])]()
            
            for (index, sentence) in sentences.enumerated() {
                sentenceGroup.enter()
                sentenceQueue.async {
                    if let embedding = self.getCachedOrComputeEmbedding(for: sentence) {
                        sentenceQueue.async(flags: .barrier) {
                            tempEmbeddings.append((index, sentence, embedding))
                        }
                    }
                    sentenceGroup.leave()
                }
            }
            
            sentenceGroup.wait()
            
            // Sort by original index to maintain order
            tempEmbeddings.sort { $0.0 < $1.0 }
            sentenceEmbeddings = tempEmbeddings.map { ($0.1, $0.2) }
            
            document.sentenceEmbeddings = sentenceEmbeddings
        }
        
        document.isProcessed = true
    }
    
    // MARK: - Search Methods with Caching
    
    func searchRelevantSentences(for query: String, limit: Int = 3) -> [(documentID: String, text: String, similarity: Double)] {
        guard let queryEmbedding = getCachedOrComputeEmbedding(for: query) else { return [] }
        
        var scored: [(String, String, Double)] = []
        
        documentQueue.sync {
            for doc in documents {
                guard doc.isProcessed else { continue }
                
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
    
    // MARK: - Enhanced Embedding Methods with Caching
    
    private func getCachedOrComputeEmbedding(for text: String) -> [Double]? {
        // Check cache first
        if let cachedEmbedding = embeddingCache.getCachedEmbedding(for: text) {
            return cachedEmbedding
        }
        
        // Compute embedding
        if let embedding = getEnhancedEmbedding(for: text) {
            // Cache the result
            embeddingCache.cacheEmbedding(embedding, for: text)
            return embedding
        }
        
        return nil
    }
    
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
    
    // MARK: - Cache Management
    
    func clearCache() {
        embeddingCache.clearCache()
    }
    
    func preloadDocuments() async {
        // Ensure all documents are processed
        await withTaskGroup(of: Void.self) { group in
            for document in documents where !document.isProcessed {
                group.addTask {
                    await withCheckedContinuation { continuation in
                        self.processingQueue.async {
                            self.processDocument(document)
                            continuation.resume()
                        }
                    }
                }
            }
        }
    }
}
