
import UIKit
import Flutter
import SwiftUI
import Foundation
import CoreML
import os
import flwr


@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    
    public var scenarioSelection = Constants.ScenarioTypes.MNIST {
        didSet {
            self.resetPreperation()
        }
    }
    public var trainingBatchStatus = Constants.PreparationStatus.notPrepared
    let scenarios = Constants.ScenarioTypes.allCases
    public var benchmarkSuite = BenchmarkSuite.shared
    private var trainingBatchProvider: MLBatchProvider?
    public var testBatchStatus = Constants.PreparationStatus.notPrepared
    private var testBatchProvider: MLBatchProvider?
    public var mlFlwrClientStatus = Constants.PreparationStatus.notPrepared
    private var mlFlwrClient: MLFlwrClient?
    private var localClient: LocalClient?
    public var localClientStatus = Constants.PreparationStatus.notPrepared
    public var federatedServerStatus = Constants.TaskStatus.idle
    private var flwrGRPC: FlwrGRPC?
    private var eventSink: FlutterEventSink?
    public var hostname: String = "ccl.gachon.ac.kr"
    public var port: Int = 40021
    
    
    public var epoch: Int = 5
    public var localTrainingStatus = Constants.TaskStatus.idle
    public var localTestStatus = Constants.TaskStatus.idle
    
    func setEventSink(_ sink: FlutterEventSink?) {
            self.eventSink = sink
        }

        // Use eventSink when you need to send events
        private func sendEvent(message: String) {
            eventSink?(message)
        }
    
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
      GeneratedPluginRegistrant.register(with: self)
      
      let controller : FlutterViewController = window?.rootViewController as! FlutterViewController
     
      let methodChannel = FlutterMethodChannel(name: "grpcCall", binaryMessenger: controller.binaryMessenger)
      let eventChannel = FlutterEventChannel(name: "TrainingListenerJavaToFlutter", binaryMessenger: controller.binaryMessenger)
      
      eventChannel.setStreamHandler(self)
      
      methodChannel.setMethodCallHandler({ [weak self] (call: FlutterMethodCall, result: @escaping FlutterResult) -> Void in
        
          print("FedOpsStarted")
          
          
          guard let strongSelf = self else {
          result(FlutterError(code: "yourErrorCode", message: "Self is nil", details: nil))
          return
        }
        
        switch call.method {
        case "loadTrainDataset":
            strongSelf.prepareTrainDataset { success, error in
                if success {
                    result("Train dataset is prepared" )
                      } else {
                          result(FlutterError(code: "prepareTrainDatasetError", message: error, details: nil))
                      }
                    }
        case "loadTestDataset":
            strongSelf.prepareTestDataset() { success, error in
                if success {
                    result("Test dataset is prepared" )
                      } else {
                          result(FlutterError(code: "prepareTrainDatasetError", message: error, details: nil))
                      }
                    }
        case "connect":
            strongSelf.initMLFlwrClient() { success, error in
                if success {
                    result("Ready to train" )
                      } else {
                          result(FlutterError(code: "prepareTrainDatasetError", message: error, details: nil))
                      }
                    }
//        case "startTraining":
//            strongSelf.startFederatedLearning()
        default:
          result(FlutterMethodNotImplemented)
        }
      })

      return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
    public func resetPreperation() {
        self.trainingBatchStatus = .notPrepared
        self.testBatchStatus = .notPrepared
        self.localClientStatus = .notPrepared
        self.mlFlwrClientStatus = .notPrepared
    }
    public func prepareTrainDataset(completion: @escaping (Bool, String?) -> Void) {
        trainingBatchStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "preparing train dataset " + scenarioSelection.description))
        DispatchQueue.global(qos: .userInitiated).async {
            let batchProvider = DataLoader.trainBatchProvider(scenario: self.scenarioSelection) { count in
                DispatchQueue.main.async {
                    self.trainingBatchStatus = .preparing(count: count)
                }
            }
            DispatchQueue.main.async {
                self.trainingBatchProvider = batchProvider
                print("prepareTrainDataset - finish")
                print(self.trainingBatchStatus.description)
                self.trainingBatchStatus = .ready
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finished preparing train dataset"))
                completion(true, nil)
            }
            
        }
    }
    public func prepareTestDataset(completion: @escaping (Bool, String?) -> Void) {
        testBatchStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "preparing test dataset " + scenarioSelection.description))
        DispatchQueue.global(qos: .userInitiated).async {
            let batchProvider = DataLoader.testBatchProvider(scenario: self.scenarioSelection) { count in
                DispatchQueue.main.async {
                    self.testBatchStatus = .preparing(count: count)
                }
            }
            DispatchQueue.main.async {
                self.testBatchProvider = batchProvider
                print("prepareTestDataset - finish")
                print(self.testBatchStatus.description)
                self.testBatchStatus = .ready
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finished test dataset"))
                completion(true, nil)
            }
            
        }
    }
    
    public func initMLFlwrClient(completion: @escaping (Bool, String?) -> Void) {
        self.mlFlwrClientStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "init ML Flwr Client with " + scenarioSelection.modelName))
        if self.mlFlwrClient == nil {
        
            DispatchQueue.global(qos: .userInitiated).async {
                let dataLoader = MLDataLoader(trainBatchProvider: self.trainingBatchProvider!, testBatchProvider: self.testBatchProvider!)
              
                if let modelUrl = Bundle.main.url(forResource:self.scenarioSelection.modelName, withExtension: "mlmodel") {
                    
                    self.initClient(modelUrl: modelUrl, dataLoader: dataLoader, clientType: .federated)
                    DispatchQueue.main.async {
                        self.mlFlwrClientStatus = .ready
                        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "ML Flwr Client ready"))
                        completion(true, nil)
                    }
                    
                }else{
                    print("Error occured during finding model file")
                }
                
            }
        } else {
            self.mlFlwrClientStatus = .ready
            self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "ML Flwr Client ready"))
            print("initMLFlwrClient - finish")
        }
    }
    
    private func initClient(modelUrl url: URL, dataLoader: MLDataLoader, clientType: Constants.ClientType) {
        do {
            let compiledModelUrl = try MLModel.compileModel(at: url)
            switch clientType {
            case .federated:
                 let modelInspect = try MLModelInspect(serializedData: Data(contentsOf: url))
                let layerWrappers = modelInspect.getLayerWrappers()
                self.mlFlwrClient = MLFlwrClient(layerWrappers: layerWrappers,
                                                 dataLoader: dataLoader,
                                                 compiledModelUrl: compiledModelUrl,
                                                 modelUrl: url)
                
                

            case .local:
                self.localClient = LocalClient(dataLoader: dataLoader, compiledModelUrl: compiledModelUrl)
            }
            
        } catch {}
    }
    
    public func startFederatedLearning() {
        guard let eventSink = self.eventSink else {
               // Handle the case where eventSink is nil
               return
           }
        self.federatedServerStatus = .ongoing(info: "Starting federated learning")
        self.eventSink?("Fl started")
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "starting federated learning"))
        if self.flwrGRPC == nil {
            
            self.flwrGRPC = FlwrGRPC(serverHost: hostname, serverPort: port, extendedInterceptor: BenchmarkInterceptor())
        }
        
    
        self.flwrGRPC?.startFlwrGRPC(client: self.mlFlwrClient!) {
            
            DispatchQueue.main.async {
                self.eventSink?("FL is finished")
                self.federatedServerStatus = .completed(info: "Federated learning completed")
                self.flwrGRPC = nil
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "Federated learning completed"))
            }
        }
    }


}

extension AppDelegate: FlutterStreamHandler {
    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
     
        self.eventSink = events
        self.mlFlwrClient?.setEventSink(self.eventSink)
        self.startFederatedLearning()
        print("we started listening")
        return nil
    }

    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        // Cleanup when the stream is canceled
        self.eventSink = nil
        return nil
    }
}

class LocalClient {
    private var dataLoader: MLDataLoader
    private var compiledModelUrl: URL
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: LocalClient.self))
    
    init(dataLoader: MLDataLoader, compiledModelUrl: URL) {
        self.dataLoader = dataLoader
        self.compiledModelUrl = compiledModelUrl
    }
    
    func runMLTask(statusHandler: @escaping (Constants.TaskStatus) -> Void,
                   numEpochs: Int,
                   task: MLTask
    ) {
        let dataset: MLBatchProvider
        let configuration = MLModelConfiguration()
        let epochs = MLParameterKey.epochs
        configuration.parameters = [epochs:numEpochs]
       
        switch task {
        case .train:
            dataset = self.dataLoader.trainBatchProvider
        case .test:
            dataset = self.dataLoader.testBatchProvider
        }
        
        var startTime = Date()
        let progressHandler = { (contextProgress: MLUpdateContext) in
            switch contextProgress.event {
            case .trainingBegin:
                let taskStatus: Constants.TaskStatus = .ongoing(info: "Started to \(task) locally")
                statusHandler(taskStatus)
            case .epochEnd:
                let taskStatus: Constants.TaskStatus
                let loss = String(format: "%.4f", contextProgress.metrics[.lossValue] as! Double)
                switch task {
                case .train:
                    let epochIndex = contextProgress.metrics[.epochIndex] as! Int
                    taskStatus = .ongoing(info: "Epoch \(epochIndex + 1) end with loss \(loss)")
                case .test:
                    taskStatus = .ongoing(info: "Local test end with loss \(loss)")
                }
                statusHandler(taskStatus)
            default:
                self.log.info("Unknown event")
            }
        }
        
        let completionHandler = { (finalContext: MLUpdateContext) in
            let loss = String(format: "%.4f", finalContext.metrics[.lossValue] as! Double)
            let taskStatus: Constants.TaskStatus = .completed(info: "Local \(task) completed with loss: \(loss) in \(Int(Date().timeIntervalSince(startTime))) secs")
            statusHandler(taskStatus)
        }
        
        
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.trainingBegin, .epochEnd],
            progressHandler: progressHandler,
            completionHandler: completionHandler
        )
        
        startTime = Date()
        do {
            let updateTask = try MLUpdateTask(forModelAt: compiledModelUrl,
                                              trainingData: dataset,
                                              configuration: configuration,
                                              progressHandlers: progressHandlers)
            updateTask.resume()
            
        } catch let error {
            log.error("\(error)")
        }
    }
}
