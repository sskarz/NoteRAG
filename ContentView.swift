//
//  ContentView.swift
//  NoteRAG
//
//  Created by Sanskar Thapa on 7/14/25.
//

import SwiftUI
import PDFKit
import UniformTypeIdentifiers
import Combine
// MARK: - Models

struct Note: Identifiable, Codable {
    let id = UUID()
    var title: String
    var content: String
    let createdAt = Date()
    
    var displayTitle: String {
        title.isEmpty ? "Untitled Note" : title
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    let timestamp = Date()
}

// MARK: - View Model

@MainActor
class NoteRAGViewModel: ObservableObject {
    @Published var notes: [Note] = []
    @Published var uploadedDocuments: [String] = []
    @Published var chatMessages: [ChatMessage] = []
    @Published var isProcessing = false
    @Published var currentQuery = ""
    
    private let rag = RAG()
    private var documentsProcessed = false
    
    init() {
        loadNotes()
    }
    
    // MARK: - Notes Management
    
    func addNote(_ note: Note) {
        notes.append(note)
        saveNotes()
        
        // Add to RAG immediately
        Task {
            let document = Document(id: "note-\(note.id.uuidString)", content: "\(note.title). \(note.content)")
            await rag.addDocument(document)
        }
    }
    
    func updateNote(_ note: Note) {
        if let index = notes.firstIndex(where: { $0.id == note.id }) {
            notes[index] = note
            saveNotes()
            
            // Update in RAG (for simplicity, we'll re-process all notes)
            Task {
                await reprocessAllDocuments()
            }
        }
    }
    
    func deleteNote(_ note: Note) {
        notes.removeAll { $0.id == note.id }
        saveNotes()
        
        // Reprocess RAG
        Task {
            await reprocessAllDocuments()
        }
    }
    
    private func saveNotes() {
        if let encoded = try? JSONEncoder().encode(notes) {
            UserDefaults.standard.set(encoded, forKey: "saved_notes")
        }
    }
    
    private func loadNotes() {
        if let data = UserDefaults.standard.data(forKey: "saved_notes"),
           let decoded = try? JSONDecoder().decode([Note].self, from: data) {
            notes = decoded
        }
    }
    
    // MARK: - Document Upload
    
    func uploadPDF(url: URL) async {
        guard let pdfDocument = PDFDocument(url: url) else { return }
        
        var fullText = ""
        for i in 0..<pdfDocument.pageCount {
            if let page = pdfDocument.page(at: i),
               let pageContent = page.string {
                fullText += pageContent + "\n"
            }
        }
        
        if !fullText.isEmpty {
            let fileName = url.lastPathComponent
            uploadedDocuments.append(fileName)
            
            let document = Document(id: "pdf-\(url.lastPathComponent)", content: fullText)
            await rag.addDocument(document)
        }
    }
    
    // MARK: - RAG Processing
    
    private func reprocessAllDocuments() async {
        // Clear cache and reprocess
        rag.clearCache()
        
        // Create documents for all notes
        var documents: [Document] = []
        
        for note in notes {
            let document = Document(
                id: "note-\(note.id.uuidString)",
                content: "\(note.title). \(note.content)"
            )
            documents.append(document)
        }
        
        // Add all documents
        await rag.addDocuments(documents)
        documentsProcessed = true
    }
    
    func ensureDocumentsProcessed() async {
        if !documentsProcessed {
            await reprocessAllDocuments()
        }
    }
    
    // MARK: - Chat Interface
    
    func sendMessage() async {
        let query = currentQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return }
        
        // Add user message
        chatMessages.append(ChatMessage(text: query, isUser: true))
        currentQuery = ""
        isProcessing = true
        
        // Ensure documents are processed
        await ensureDocumentsProcessed()
        
        do {
            // Get response from RAG
            let response = try await rag.generateResponse(for: query)
            
            // Add AI response
            chatMessages.append(ChatMessage(text: response, isUser: false))
        } catch {
            chatMessages.append(ChatMessage(text: "Sorry, I couldn't process your question. Error: \(error.localizedDescription)", isUser: false))
        }
        
        isProcessing = false
    }
}

// MARK: - Main App View

struct ContentView: View {
    @StateObject private var viewModel = NoteRAGViewModel()
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            NotesView(viewModel: viewModel)
                .tabItem {
                    Label("Notes", systemImage: "note.text")
                }
                .tag(0)
            
            ChatView(viewModel: viewModel)
                .tabItem {
                    Label("Ask AI", systemImage: "bubble.left.and.bubble.right")
                }
                .tag(1)
        }
    }
}

// MARK: - Notes View

struct NotesView: View {
    @ObservedObject var viewModel: NoteRAGViewModel
    @State private var showingNewNote = false
    @State private var showingDocumentPicker = false
    
    var body: some View {
        NavigationView {
            List {
                Section("My Notes") {
                    ForEach(viewModel.notes) { note in
                        NavigationLink(destination: NoteEditorView(viewModel: viewModel, note: note)) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(note.displayTitle)
                                    .font(.headline)
                                Text(note.content)
                                    .font(.caption)
                                    .lineLimit(2)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .onDelete { indexSet in
                        indexSet.forEach { index in
                            viewModel.deleteNote(viewModel.notes[index])
                        }
                    }
                }
                
                if !viewModel.uploadedDocuments.isEmpty {
                    Section("Uploaded Documents") {
                        ForEach(viewModel.uploadedDocuments, id: \.self) { document in
                            HStack {
                                Image(systemName: "doc.fill")
                                    .foregroundColor(.blue)
                                Text(document)
                                    .font(.subheadline)
                            }
                        }
                    }
                }
            }
            .navigationTitle("NoteRAG")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(action: { showingNewNote = true }) {
                            Label("New Note", systemImage: "square.and.pencil")
                        }
                        
                        Button(action: { showingDocumentPicker = true }) {
                            Label("Upload PDF", systemImage: "doc.badge.plus")
                        }
                    } label: {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showingNewNote) {
                NavigationView {
                    NoteEditorView(viewModel: viewModel, note: nil)
                }
            }
            .fileImporter(
                isPresented: $showingDocumentPicker,
                allowedContentTypes: [.pdf],
                allowsMultipleSelection: false
            ) { result in
                Task {
                    if case .success(let urls) = result,
                       let url = urls.first {
                        await viewModel.uploadPDF(url: url)
                    }
                }
            }
        }
    }
}

// MARK: - Note Editor View

struct NoteEditorView: View {
    @ObservedObject var viewModel: NoteRAGViewModel
    @Environment(\.presentationMode) var presentationMode
    
    @State private var title: String
    @State private var content: String
    private let note: Note?
    
    init(viewModel: NoteRAGViewModel, note: Note?) {
        self.viewModel = viewModel
        self.note = note
        self._title = State(initialValue: note?.title ?? "")
        self._content = State(initialValue: note?.content ?? "")
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section("Title") {
                    TextField("Note title", text: $title)
                }
                
                Section("Content") {
                    TextEditor(text: $content)
                        .frame(minHeight: 200)
                }
            }
            .navigationTitle(note == nil ? "New Note" : "Edit Note")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveNote()
                    }
                    .disabled(title.isEmpty && content.isEmpty)
                }
            }
        }
    }
    
    private func saveNote() {
        if let existingNote = note {
            var updatedNote = existingNote
            updatedNote.title = title
            updatedNote.content = content
            viewModel.updateNote(updatedNote)
        } else {
            let newNote = Note(title: title, content: content)
            viewModel.addNote(newNote)
        }
        presentationMode.wrappedValue.dismiss()
    }
}

// MARK: - Chat View

struct ChatView: View {
    @ObservedObject var viewModel: NoteRAGViewModel
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Chat messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(viewModel.chatMessages) { message in
                                ChatBubbleView(message: message)
                                    .id(message.id)
                            }
                            
                            if viewModel.isProcessing {
                                HStack {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                    Text("Thinking...")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .padding()
                            }
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.chatMessages.count) { _ in
                        withAnimation {
                            proxy.scrollTo(viewModel.chatMessages.last?.id, anchor: .bottom)
                        }
                    }
                }
                
                Divider()
                
                // Input area
                HStack(spacing: 12) {
                    TextField("Ask about your notes...", text: $viewModel.currentQuery)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .focused($isInputFocused)
                        .onSubmit {
                            Task {
                                await viewModel.sendMessage()
                            }
                        }
                    
                    Button(action: {
                        Task {
                            await viewModel.sendMessage()
                        }
                    }) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundColor(.blue)
                    }
                    .disabled(viewModel.currentQuery.isEmpty || viewModel.isProcessing)
                }
                .padding()
                .background(Color(UIColor.secondarySystemBackground))
            }
            .navigationTitle("Ask AI")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        viewModel.chatMessages.removeAll()
                    }) {
                        Image(systemName: "trash")
                    }
                    .disabled(viewModel.chatMessages.isEmpty)
                }
            }
        }
    }
}

// MARK: - Chat Bubble View

struct ChatBubbleView: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser { Spacer() }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .padding(12)
                    .background(message.isUser ? Color.blue : Color(UIColor.secondarySystemFill))
                    .foregroundColor(message.isUser ? .white : .primary)
                    .cornerRadius(16)
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: UIScreen.main.bounds.width * 0.75, alignment: message.isUser ? .trailing : .leading)
            
            if !message.isUser { Spacer() }
        }
    }
}
