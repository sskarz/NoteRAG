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
    
    rag.addDocument(Document(id: "1", content: "Swift is a programming language developed by Apple for iOS, macOS, watchOS, and tvOS."))
    rag.addDocument(Document(id: "2", content: "Swift was designed to be safer and more concise than Objective-C, with modern features."))
    rag.addDocument(Document(id: "3", content: "Key features of Swift include type safety, type inference, and automatic memory management."))
    rag.addDocument(Document(id: "4", content: "My dog ate the homework, and my cat is very cute."))
    rag.addDocument(Document(id: "5", content: "Taylor Swift is a very cool artist and her music is amazing! I wish to see her in concert one day!"))
    rag.addDocument(Document(id: "6", content: "To make great pancakes, mix flour, eggs, milk, and a pinch of salt. Cook on a hot griddle until golden brown on both sides."))
    rag.addDocument(Document(id: "7", content: "Japan is known for its cherry blossoms, ancient temples, and delicious sushi cuisine."))
    rag.addDocument(Document(id: "8", content: "A to-do list app helps users organize daily tasks and boost productivity by setting reminders and priorities."))
    rag.addDocument(Document(id: "9", content: "Rainforests are vital ecosystems that are home to a vast diversity of plant and animal species."))
    rag.addDocument(Document(id: "10", content: "The internet of things (IoT) connects everyday devices to the internet, enabling smarter homes and cities."))
    
    let query = "What is that term called that connects online devices?"
    let response = try await rag.generateResponse(for: query)
    
    print("Question: \(query)")
    print("Answer: \(response)")

}

