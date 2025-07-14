//
//  TestModelRAG.swift
//  NoteRag
//
//  Created by Sanskar Thapa on 7/14/25.
//

import Foundation
import Playgrounds

#Playground {
    let rag = RAG()
    
    // Create documents
    let documents = [
        Document(id: "1", content: "Swift is a powerful and intuitive programming language developed by Apple for building apps for iOS, macOS, watchOS, and tvOS. It was introduced in 2014 as a modern replacement for Objective-C."),
        Document(id: "2", content: "Swift was designed to be safer and more concise than Objective-C, with modern features like optionals, type inference, and protocol-oriented programming. It eliminates entire classes of unsafe code."),
        Document(id: "3", content: "Key features of Swift include type safety, type inference, automatic memory management through ARC, closures, generics, and powerful error handling. Swift also supports functional programming patterns."),
        Document(id: "4", content: "My dog ate the homework, and my cat is very cute. This has nothing to do with programming."),
        Document(id: "5", content: "Taylor Swift is a very cool artist and her music is amazing! I wish to see her in concert one day!"),
        Document(id: "6", content: "To make great pancakes, mix flour, eggs, milk, and a pinch of salt. Cook on a hot griddle until golden brown on both sides."),
        Document(id: "7", content: "Swift's memory management uses Automatic Reference Counting (ARC) which automatically frees up memory used by class instances when they're no longer needed. This prevents memory leaks while maintaining performance."),
        Document(id: "8", content: "Protocol-oriented programming is a key paradigm in Swift. Protocols define a blueprint of methods, properties, and requirements that suit a particular task or piece of functionality."),
        Document(id: "9", content: "Swift playgrounds provide an interactive environment for learning and experimenting with Swift code. They're perfect for prototyping and educational purposes."),
        Document(id: "10", content: "The internet of things (IoT) connects everyday devices to the internet, enabling smarter homes and cities."),
        Document(id: "11", content: "SwiftUI is Apple's modern declarative framework for building user interfaces across all Apple platforms using Swift code."),
        Document(id: "12", content: "Pizza originated in Italy and is loved worldwide for its crispy crust, melted cheese, and endless topping combinations."),
        Document(id: "13", content: "Swift Package Manager is a tool for managing the distribution of Swift code. It's integrated with the Swift build system to automate downloading, compiling, and linking dependencies."),
        Document(id: "14", content: "Soccer, or football, is the world's most popular sport and is played and watched by millions globally."),
        Document(id: "15", content: "Swift's strong typing system helps catch errors at compile time rather than runtime, making apps more stable and reliable.")
    ]
    
    // Add all documents asynchronously
    await rag.addDocuments(documents)
    
    // Query and generate response
    let query = "What are some features of the programming language Swift that would convince me to switch from programming in Kotlin?"
    
    do {
        let response = try await rag.generateResponse(for: query)
        print("Question: \(query)")
        print("\nAnswer: \(response)")
    } catch {
        print("Error: \(error)")
    }
}
