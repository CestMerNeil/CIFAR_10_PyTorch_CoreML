//
//  main.swift
//  CIFAR10MPS_Swift
//
//  Created by Ao XIE on 13/09/2024.
//

import Foundation
import CoreML
import Vision

// Function to load the image from the project folder
func getImage(image: String) -> CGImage? {
    // Construct the path to the image in the project folder
    let imagePath = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .appendingPathComponent("images/\(image)").path

    print("Image Path: \(imagePath)")
    let imageURL = URL(fileURLWithPath: imagePath)
    
    // Check if the file exists at the specified path
    guard FileManager.default.fileExists(atPath: imageURL.path) else {
        print("File does not exist: \(imageURL.path)")
        return nil
    }
    
    // Create a CGImageSource from the image URL
    guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil) else {
        print("Cannot create image source.")
        return nil
    }
    
    // Return the CGImage at index 0
    return CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
}

// Function to load the CoreML model
func loadModel() -> VNCoreMLModel? {
    do {
        // Load the CoreML model into a Vision model for image classification
        return try VNCoreMLModel(for: CIFAR10NetMPS_4().model)
    } catch {
        fatalError("Unable to load model: \(error)")
    }
}

// Function to compute softmax on an array of logits
func softmax(_ logits: [Float]) -> [Float] {
    // Calculate exponentials of all logits
    let expValues = logits.map { exp($0) }
    
    // Sum of all exponential values
    let sumExpValues = expValues.reduce(0, +)
    
    // Normalize each exponential value by the sum of all exponential values
    return expValues.map { $0 / sumExpValues }
}

// Function to make a prediction using the provided image and model
func predictImage(image: CGImage, model: VNCoreMLModel) {
    // Create a Vision request for image classification using the provided model
    let request = VNCoreMLRequest(model: model) { (request, error) in
        if let results = request.results as? [VNClassificationObservation] {
            // 1. Collect all confidence values for the classifications
            let confidences = results.map { $0.confidence }
            
            // 2. Calculate softmax on the confidence values
            let softmaxConfidences = softmax(confidences)
            
            // 3. Iterate over results and print normalized confidence (Softmax) for each classification
            for (index, result) in results.enumerated() {
                let softmaxConfidence = softmaxConfidences[index]
                print("Classification: \(result.identifier), Softmax confidence: \(softmaxConfidence)")
            }

            // Print the classification with the highest confidence
            if let bestResultIndex = softmaxConfidences.enumerated().max(by: { a, b in a.element < b.element })?.offset {
                let bestResult = results[bestResultIndex]
                print("Top classification: \(bestResult.identifier), Softmax confidence: \(softmaxConfidences[bestResultIndex])")
            }
        } else {
            print("Unable to get results: \(error?.localizedDescription ?? "Unknown error")")
        }
    }

    // Create a Vision handler to process the image and perform the classification request
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        // Perform the request
        try handler.perform([request])
    } catch {
        print("Inference failed: \(error)")
    }
}

// Main function to load the image, model, and make a prediction
if let image = getImage(image: "airplane.jpg") {
    if let model = loadModel() {
        predictImage(image: image, model: model)
    }
}
