package flwr.android_client;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.Resources;
import android.graphics.drawable.Drawable;
import android.icu.text.SimpleDateFormat;
import android.nfc.Tag;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.util.Patterns;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import flwr.android_client.FlowerServiceGrpc.FlowerServiceStub;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.google.protobuf.ByteString;

import io.grpc.stub.StreamObserver;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    private EditText ip;
    private EditText port;
    private Button loadDataButton;
    private Button connectButton;
    private Button trainButton;
    private TextView resultText;

    private EditText device_id;
    private ManagedChannel channel;
    public FlowerClient fc;
    private static final String TAG = "Flower";

    public ProgressBar mProgress;

    public TextView tv;
    public TextView roundText;
    public TextView epochText;
    public TextView lossText;
    public TextView accuracyText;
    Handler handler;
    public List<Float> lossList;
    public LineChart lineChart;
    public LineChart lineChart1;

    public List<Float> accuracyList;
    public int localRound = 1;

    public int localEpoch = 1;
    public int round_epochs = 0;
    public int training_rounds = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        resultText = (TextView) findViewById(R.id.grpc_response_text);
        resultText.setMovementMethod(new ScrollingMovementMethod());
        device_id = (EditText) findViewById(R.id.device_id_edit_text);
        ip = (EditText) findViewById(R.id.serverIP);
        port = (EditText) findViewById(R.id.serverPort);
        loadDataButton = (Button) findViewById(R.id.load_data);
        connectButton = (Button) findViewById(R.id.connect);
        trainButton = (Button) findViewById(R.id.trainFederated);
        Resources res = getResources();
        Drawable drawable = res.getDrawable(R.drawable.circle_process);
        mProgress = (ProgressBar) findViewById(R.id.circularProgressbar);
        mProgress.setProgress(0);   // Main Progress
        mProgress.setSecondaryProgress(100); // Secondary Progress
        mProgress.setMax(100); // Maximum Progress
        mProgress.setProgressDrawable(drawable);
        tv = (TextView) findViewById(R.id.tv);
        mProgress.getProgress();
        handler = new Handler();
        roundText = (TextView) findViewById(R.id.round_txt);
        epochText = (TextView) findViewById(R.id.epoch_txt);
        lossText = (TextView) findViewById(R.id.loss_txt);
        accuracyText = (TextView) findViewById(R.id.accuracy_txt);
        accuracyList = new ArrayList<>();
        lossList = new ArrayList<>();
        lineChart = findViewById(R.id.lineChart);
        lineChart1 = findViewById(R.id.lineChart1);


        new Thread(new Runnable() {
            @Override
            public void run() {

                while (mProgress.getProgress() <= 100) {
                    handler.post(new Runnable() {
                        @SuppressLint("SetTextI18n")
                        @Override
                        public void run() {
                            tv.setText(mProgress.getProgress() + " %");
//                            epochText.setText(String.valueOf((mProgress.getProgress() * training_rounds * round_epochs) / 100 - (localRound - 1) * round_epochs));
                        }
                    });
                    try {
                        Thread.sleep(300);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }).start();

        fc = new FlowerClient(this);
    }

    public static void hideKeyboard(Activity activity) {
        InputMethodManager imm = (InputMethodManager) activity.getSystemService(Activity.INPUT_METHOD_SERVICE);
        View view = activity.getCurrentFocus();
        if (view == null) {
            view = new View(activity);
        }
        imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }
   public void setRoundText( ) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                roundText.setText("2");
                epochText.setText("2");
            }
        });

    }

    void setLossAccuracy(double accuracy, double loss) {
        accuracyText.setText(String.valueOf(accuracy));
        lossText.setText(String.valueOf(loss));
    }

    void setTableResults(List<Float> accList, List<Float> lssList) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {

                lineChart.setVisibility(View.VISIBLE);
                lineChart1.setVisibility(View.VISIBLE);
                List<Entry> entries = new ArrayList<>();
                lssList.forEach(loss -> {

                    entries.add(new Entry(lssList.indexOf(loss), loss));
                });


                List<Entry> entries1 = new ArrayList<>();

                accList.forEach(acc -> {

                    entries1.add(new Entry(accList.indexOf(acc), acc));
                });
                LineDataSet dataSet = new LineDataSet(entries, "Training loss");
                LineDataSet dataSet1 = new LineDataSet(entries, "Accuracy");

                LineData lineData = new LineData(dataSet);
                LineData lineData1 = new LineData(dataSet1);


                LineChart lineChart = findViewById(R.id.lineChart);
                lineChart.setData(lineData);
                lineChart.invalidate();

                LineChart lineChart1 = findViewById(R.id.lineChart1);
                lineChart1.setData(lineData1);
                lineChart1.invalidate();
                dataSet.notifyDataSetChanged();
                lineData.notifyDataChanged();
                lineChart.notifyDataSetChanged();
                lineChart.invalidate();
                lineChart1.notifyDataSetChanged();
                lineChart1.invalidate();
                lineData1.notifyDataChanged();
                dataSet1.notifyDataSetChanged();
            }
        });
    }


    public void setResultText(String text) {
        SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss", Locale.GERMANY);
        String time = dateFormat.format(new Date());
        resultText.append("\n" + time + "   " + text);
    }

    public void loadData(View view) {
        if (TextUtils.isEmpty(device_id.getText().toString())) {
            Toast.makeText(this, "Please enter a client partition ID between 1 and 10 (inclusive)", Toast.LENGTH_LONG).show();
        } else if (Integer.parseInt(device_id.getText().toString()) > 10 || Integer.parseInt(device_id.getText().toString()) < 1) {
            Toast.makeText(this, "Please enter a client partition ID between 1 and 10 (inclusive)", Toast.LENGTH_LONG).show();
        } else {
            hideKeyboard(this);
            setResultText("Loading the local training dataset in memory. It will take several seconds.");
            loadDataButton.setEnabled(false);

            ExecutorService executor = Executors.newSingleThreadExecutor();
            Handler handler = new Handler(Looper.getMainLooper());

            executor.execute(new Runnable() {
                private String result;

                @Override
                public void run() {
                    try {
                        fc.loadData(Integer.parseInt(device_id.getText().toString()));
                        result = "Training dataset is loaded in memory.";
                    } catch (Exception e) {
                        StringWriter sw = new StringWriter();
                        PrintWriter pw = new PrintWriter(sw);
                        e.printStackTrace(pw);
                        pw.flush();
                        result = "Training dataset is loaded in memory.";
                    }
                    handler.post(() -> {
                        setResultText(result);
                        connectButton.setEnabled(true);
                    });
                }
            });
        }
    }

    public void connect(View view) {
        String host = ip.getText().toString();
        String portStr = port.getText().toString();
        if (TextUtils.isEmpty(host) || TextUtils.isEmpty(portStr) || !Patterns.IP_ADDRESS.matcher(host).matches()) {
            Toast.makeText(this, "Please enter the correct IP and port of the FL server", Toast.LENGTH_LONG).show();
        } else {
            int port = TextUtils.isEmpty(portStr) ? 0 : Integer.parseInt(portStr);
            channel = ManagedChannelBuilder.forAddress(host, port).maxInboundMessageSize(10 * 1024 * 1024).usePlaintext().build();
            hideKeyboard(this);
            trainButton.setEnabled(true);
            connectButton.setEnabled(false);
            setResultText("Channel object created. Ready to train!");
        }
    }

    public void runGrpc(View view) {
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
                handler.post(() -> {
                    setResultText(result);
                    trainButton.setEnabled(false);
                });
            }
        });
    }

    public void restart(View view) {
        loadDataButton.setEnabled(true);
        connectButton.setEnabled(false);
        trainButton.setEnabled(false);
        port.setText("");
        ip.setText("");
        device_id.setText("");
        resultText.setText("");
        fc = null;
    }


    private static class FlowerServiceRunnable {
        protected Throwable failed;
        private StreamObserver<ClientMessage> requestObserver;

        public void run(FlowerServiceStub asyncStub, MainActivity activity) {
            join(asyncStub, activity);
        }

        private void join(FlowerServiceStub asyncStub, MainActivity activity)
                throws RuntimeException {


            final CountDownLatch finishLatch = new CountDownLatch(1);
            requestObserver = asyncStub.join(
                    new StreamObserver<ServerMessage>() {

                        @Override
                        public void onNext(ServerMessage msg) {
                            handleMessage(msg, activity);
                            Log.e(TAG, "count::" + finishLatch.getCount());

                        }

                        @Override
                        public void onError(Throwable t) {
                            activity.channel.shutdown();
                            failed = t;
                            finishLatch.countDown();
                            activity.setResultText("Finished and Connection channel closed");
                            Log.e(TAG, "" + failed);
                            activity.setTableResults(activity.accuracyList, activity.lossList);

                        }

                        @Override
                        public void onCompleted() {
                            finishLatch.countDown();

                            activity.setResultText("done");
                            Log.e(TAG, "Done");
                        }
                    });
        }

        private void handleMessage(ServerMessage message, MainActivity activity) {
            try {
                ByteBuffer[] weights;
                ClientMessage c = null;

                if (message.hasGetParametersIns()) {
                    Log.e(TAG, "Handling GetParameters");
                    activity.setResultText("Handling GetParameters message from the server.");

                    weights = activity.fc.getWeights();
                    c = weightsAsProto(weights);
                } else if (message.hasFitIns()) {
                    Log.e(TAG, "Handling FitIns");
                    activity.setResultText("Handling Fit request from the server.");

                    List<ByteString> layers = message.getFitIns().getParameters().getTensorsList();

                    Scalar epoch_config = message.getFitIns().getConfigMap().getOrDefault("local_epochs", Scalar.newBuilder().setSint64(1).build());
                    Scalar num_rounds = message.getFitIns().getConfigMap().getOrDefault("num_rounds", Scalar.newBuilder().setSint64(1).build());

                    assert epoch_config != null;
                    assert num_rounds != null;
                    activity.round_epochs = (int) epoch_config.getSint64();
                    activity.training_rounds = (int) num_rounds.getSint64();

                    ByteBuffer[] newWeights = new ByteBuffer[10];
                    for (int i = 0; i < 10; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }

                    Pair<ByteBuffer[], Integer> outputs = activity.fc.fit(newWeights, activity.round_epochs, activity.training_rounds, activity.mProgress);
                    c = fitResAsProto(outputs.first, outputs.second);
                } else if (message.hasEvaluateIns()) {
                    Log.e(TAG, "Handling EvaluateIns");
                    activity.setResultText("Handling Evaluate request from the server");

                    List<ByteString> layers = message.getEvaluateIns().getParameters().getTensorsList();
                    // Our model has 10 layers
                    ByteBuffer[] newWeights = new ByteBuffer[10];
                    for (int i = 0; i < 10; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }
                    Pair<Pair<Float, Float>, Integer> inference = activity.fc.evaluate(newWeights);

                    Float loss = inference.first.first;
                    Float accuracy = inference.first.second;
                    activity.setLossAccuracy(accuracy, loss);
                    activity.lossList.add(loss);
                    activity.accuracyList.add(accuracy);
                    activity.localRound++;
                    activity.setResultText("Test Accuracy after this round = " + accuracy);
                    int test_size = inference.second;
                    c = evaluateResAsProto(loss, test_size);

                }
                requestObserver.onNext(c);
                activity.setResultText("Response sent to the server");
            } catch (Exception e) {
                Log.e(TAG, "error");
                Log.e(TAG, e.getMessage());
            }
        }
    }

    private static ClientMessage weightsAsProto(ByteBuffer[] weights) {
        List<ByteString> layers = new ArrayList<>();
        for (ByteBuffer weight : weights) {
            layers.add(ByteString.copyFrom(weight));
        }
        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.GetParametersRes res = ClientMessage.GetParametersRes.newBuilder().setParameters(p).build();
        return ClientMessage.newBuilder().setGetParametersRes(res).build();
    }

    private static ClientMessage fitResAsProto(ByteBuffer[] weights, int training_size) {
        List<ByteString> layers = new ArrayList<>();
        for (ByteBuffer weight : weights) {
            layers.add(ByteString.copyFrom(weight));
        }
        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.FitRes res = ClientMessage.FitRes.newBuilder().setParameters(p).setNumExamples(training_size).build();
        return ClientMessage.newBuilder().setFitRes(res).build();
    }

    private static ClientMessage evaluateResAsProto(float accuracy, int testing_size) {
        ClientMessage.EvaluateRes res = ClientMessage.EvaluateRes.newBuilder().setLoss(accuracy).setNumExamples(testing_size).build();
        return ClientMessage.newBuilder().setEvaluateRes(res).build();
    }
}
