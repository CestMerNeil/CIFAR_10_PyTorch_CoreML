//
//  main.swift
//  CIFAR10MPS_Swift
//
//  Created by Ao XIE on 13/09/2024.
//

import Foundation
import CoreML
import Vision

func getImage(image: String) -> CGImage? {
    let imagePath = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .appendingPathComponent("images/\(image)").path

    print("Image Path: \(imagePath)")
    let imageURL = URL(fileURLWithPath: imagePath)
    guard FileManager.default.fileExists(atPath: imageURL.path) else {
        print("文件不存在: \(imageURL.path)")
        return nil
    }
    guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil) else {
        print("Can not create image source.")
        return nil
    }
    
    return CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
}

func loadModel() -> VNCoreMLModel? {
    do {
        return try VNCoreMLModel(for: CIFAR10NetMPS_4().model)
    } catch {
        fatalError("无法加载模型: \(error)")
    }
}

func softmax(_ logits: [Float]) -> [Float] {
    let expValues = logits.map { exp($0) }
    let sumExpValues = expValues.reduce(0, +)
    return expValues.map { $0 / sumExpValues }
}

func predictImage(image: CGImage, model: VNCoreMLModel) {
    let request = VNCoreMLRequest(model: model) { (request, error) in
        if let results = request.results as? [VNClassificationObservation] {
            // 1. 收集所有分类的置信度值
            let confidences = results.map { $0.confidence }
            
            // 2. 计算Softmax
            let softmaxConfidences = softmax(confidences)
            
            // 3. 遍历结果并输出归一化后的置信度
            for (index, result) in results.enumerated() {
                let softmaxConfidence = softmaxConfidences[index]
                print("分类: \(result.identifier), Softmax置信度: \(softmaxConfidence)")
            }

            // 如果你想要输出置信度最高的分类，可以使用以下代码：
            if let bestResultIndex = softmaxConfidences.enumerated().max(by: { a, b in a.element < b.element })?.offset {
                let bestResult = results[bestResultIndex]
                print("最高置信度分类: \(bestResult.identifier), Softmax置信度: \(softmaxConfidences[bestResultIndex])")
            }
        } else {
            print("无法获得结果: \(error?.localizedDescription ?? "未知错误")")
        }
    }

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
    } catch {
        print("推理失败: \(error)")
    }
}

// 主函数调用
if let image = getImage(image: "airplane.jpg") {
    if let model = loadModel() {
        predictImage(image: image, model: model)
    }
}
