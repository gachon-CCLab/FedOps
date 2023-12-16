package com.example.grpc_java;

import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.util.Log;
import android.util.Pair;

import androidx.annotation.NonNull;

import com.google.protobuf.ByteString;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import flwr.android_client.FlowerServiceGrpc;
import flwr.android_client.ServerMessage;
import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.EventChannel;
import io.flutter.plugin.common.MethodChannel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;

import android.os.Bundle;
import android.view.View;

import java.util.Objects;

public class MainActivity extends FlutterActivity {
    private ManagedChannel channel;
    private StreamObserver<flwr.android_client.ClientMessage> requestObserver;

    private static final String CHANNEL = "grpcCall";
    static String TAG = "FedOPS_Flutter";
    public FlowerClient fc;
    private ExecutorService executor = Executors.newSingleThreadExecutor();
    public int serverPort;
    public static final String STREAM = "TrainingListenerJavaToFlutter";
    public EventChannel.EventSink attachEvent;
    private Handler handler;

    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        fc = new FlowerClient(this, this);
        super.configureFlutterEngine(flutterEngine);
        new EventChannel(Objects.requireNonNull(getFlutterEngine()).getDartExecutor(), STREAM).setStreamHandler(
                new EventChannel.StreamHandler() {
                    @Override
                    public void onListen(Object arguments, EventChannel.EventSink events) {
                        attachEvent = events;
                        handler = new Handler();
                    }

                    @Override
                    public void onCancel(Object arguments) {

                    }
                }

        );
        new MethodChannel(flutterEngine.getDartExecutor().getBinaryMessenger(), CHANNEL)
                .setMethodCallHandler(
                        (call, result) -> {
                            if (call.method.equals("connect")) {
                                connect();
                                result.success("Connected to server");
                            }
                            if (call.method.equals("startTraining")) {
                                runGrpc();
                            }
                            if (call.method.equals("loadDataset")) {
                                serverPort =call.argument("serverPort");

                                boolean res = loadData();
                                if (res) {
                                    result.success("Dataset loaded to memory");
                                }
                            }
                        }
                );
    }


    public void connect() {
        String host = "ccl.gachon.ac.kr";
//        String host = "210.102.181.156";
//        String portStr = "40015";

        int port =  serverPort;
//        int port =  40015;
        channel = ManagedChannelBuilder.forAddress(host, port).maxInboundMessageSize(1024 * 1024 * 1024).usePlaintext().build();

    }

    public boolean loadData() {

        Future<Boolean> future = executor.submit(new Callable<Boolean>() {
            @Override
            public Boolean call() throws Exception {
                try {
                    fc.loadData(1);
                    return true;
                } catch (Exception e) {
                    e.printStackTrace();
                    return false;
                }
            }
        });

        try {
            // Wait for the future to complete
            return future.get();
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    public void sendMessagesToFlutter(String message) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                attachEvent.success(message);

            }
        });

    }

    public void runGrpc() {
        MainActivity activity = this;
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Handler handler = new Handler(Looper.getMainLooper());


        executor.execute(new Runnable() {
            private String result;

            @Override
            public void run() {
                try {
                    (new FlowerServiceRunnable()).run(FlowerServiceGrpc.newStub(channel), activity);
                    result = "Connection to the FL server successful \n";
                } catch (Exception e) {
                    StringWriter sw = new StringWriter();
                    PrintWriter pw = new PrintWriter(sw);
                    e.printStackTrace(pw);
                    pw.flush();
                    result = "Failed to connect to the FL server \n" + sw;
                }

            }
        });
    }

    private class FlowerServiceRunnable {
        protected Throwable failed;
        private StreamObserver<flwr.android_client.ClientMessage> requestObserver;
        public void run(FlowerServiceGrpc.FlowerServiceStub asyncStub, MainActivity activity) {
            join(asyncStub, activity);
        }

        private void join(FlowerServiceGrpc.FlowerServiceStub asyncStub, MainActivity activity)
                throws RuntimeException {


            final CountDownLatch finishLatch = new CountDownLatch(1);
            requestObserver = asyncStub.join(
                    new StreamObserver<ServerMessage>() {

                        @Override
                        public void onNext(ServerMessage msg) {
                            handleMessage(msg, activity);
                        }

                        @Override
                        public void onError(Throwable t) {
                            activity.channel.shutdown();
                            failed = t;
                            finishLatch.countDown();
                            android.util.Log.e(TAG, "" + failed);

                        }

                        @Override
                        public void onCompleted() {
                            finishLatch.countDown();
                            android.util.Log.e(TAG, "Done");
                        }
                    });
        }

        private void handleMessage(ServerMessage message, MainActivity activity) {
            try {
                ByteBuffer[] weights;
                flwr.android_client.ClientMessage c = null;
                if (message.hasFitIns()) {
                    List<ByteString> layers = message.getFitIns().getParameters().getTensorsList();
                    flwr.android_client.Scalar epoch_config = null;
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        epoch_config = message.getFitIns().getConfigMap().getOrDefault("local_epochs", flwr.android_client.Scalar.newBuilder().setSint64(1).build());
                    }
                    flwr.android_client.Scalar num_rounds = null;
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                        num_rounds = message.getFitIns().getConfigMap().getOrDefault("num_rounds", flwr.android_client.Scalar.newBuilder().setSint64(1).build());
                    }
                    assert epoch_config != null;
                    assert num_rounds != null;

                    ByteBuffer[] newWeights = new ByteBuffer[10];
                    for (int i = 0; i < 10; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }
//                    Log.e(TAG, ( num_rounds.getSint64()).toString());
                    activity.sendMessagesToFlutter("overallRounds"+(int) num_rounds.getSint64());
                    activity.sendMessagesToFlutter("overallEpochs"+(int) epoch_config.getSint64());
                    Pair<ByteBuffer[], Integer> outputs = activity.fc.fit(newWeights, (int) epoch_config.getSint64());
                    c = fitResAsProto(outputs.first, outputs.second);
                } else if (message.hasGetParametersIns()) {


                    weights = activity.fc.getWeights();
                    c = weightsAsProto(weights);
                } else if (message.hasEvaluateIns()) {
                    android.util.Log.e(TAG, "Handling EvaluateIns");
                    List<ByteString> layers = message.getEvaluateIns().getParameters().getTensorsList();
                    ByteBuffer[] newWeights = new ByteBuffer[10];
                    for (int i = 0; i < 10; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }
                    Pair<Pair<Float, Float>, Integer> inference = activity.fc.evaluate(newWeights);

                    Float loss = inference.first.first;
                    Float accuracy = inference.first.second;
                    Log.e(TAG, "accuracy::"+accuracy);
                    Log.e(TAG, "loss::"+loss);
                    sendMessagesToFlutter("accuracy "+accuracy);
                    sendMessagesToFlutter("loss "+loss);
                    int test_size = inference.second;
                    c = evaluateResAsProto(loss, test_size);
                    sendMessagesToFlutter("weights sent to server");
                }
                requestObserver.onNext(c);
            } catch (Exception e) {
                android.util.Log.e(TAG, "error");
                android.util.Log.e(TAG, e.getMessage());
                sendMessagesToFlutter(e.getMessage());
            }
        }
    }


    private static flwr.android_client.ClientMessage weightsAsProto(ByteBuffer[] weights) {
        List<ByteString> layers = new ArrayList<>();
        for (ByteBuffer weight : weights) {
            layers.add(ByteString.copyFrom(weight));
        }
        flwr.android_client.Parameters p = flwr.android_client.Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        flwr.android_client.ClientMessage.GetParametersRes res = flwr.android_client.ClientMessage.GetParametersRes.newBuilder().setParameters(p).build();
        return flwr.android_client.ClientMessage.newBuilder().setGetParametersRes(res).build();
    }

    private static flwr.android_client.ClientMessage fitResAsProto(ByteBuffer[] weights, int training_size) {
        List<ByteString> layers = new ArrayList<>();
        for (ByteBuffer weight : weights) {
            layers.add(ByteString.copyFrom(weight));
        }
        flwr.android_client.Parameters p = flwr.android_client.Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        flwr.android_client.ClientMessage.FitRes res = flwr.android_client.ClientMessage.FitRes.newBuilder().setParameters(p).setNumExamples(training_size).build();
        return flwr.android_client.ClientMessage.newBuilder().setFitRes(res).build();
    }

    private static flwr.android_client.ClientMessage evaluateResAsProto(float accuracy, int testing_size) {
        Log.e(TAG, "training results sent to server");
        flwr.android_client.ClientMessage.EvaluateRes res = flwr.android_client.ClientMessage.EvaluateRes.newBuilder().setLoss(accuracy).setNumExamples(testing_size).build();
        return flwr.android_client.ClientMessage.newBuilder().setEvaluateRes(res).build();
    }

}
