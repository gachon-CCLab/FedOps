package flwr.android_client;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.43.0)",
    comments = "Source: transport.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class FlowerServiceGrpc {

  private FlowerServiceGrpc() {}

  public static final String SERVICE_NAME = "flwr.proto.FlowerService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<flwr.android_client.ClientMessage,
      flwr.android_client.ServerMessage> getJoinMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Join",
      requestType = flwr.android_client.ClientMessage.class,
      responseType = flwr.android_client.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
  public static io.grpc.MethodDescriptor<flwr.android_client.ClientMessage,
      flwr.android_client.ServerMessage> getJoinMethod() {
    io.grpc.MethodDescriptor<flwr.android_client.ClientMessage, flwr.android_client.ServerMessage> getJoinMethod;
    if ((getJoinMethod = FlowerServiceGrpc.getJoinMethod) == null) {
      synchronized (FlowerServiceGrpc.class) {
        if ((getJoinMethod = FlowerServiceGrpc.getJoinMethod) == null) {
          FlowerServiceGrpc.getJoinMethod = getJoinMethod =
              io.grpc.MethodDescriptor.<flwr.android_client.ClientMessage, flwr.android_client.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Join"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.lite.ProtoLiteUtils.marshaller(
                  flwr.android_client.ClientMessage.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.lite.ProtoLiteUtils.marshaller(
                  flwr.android_client.ServerMessage.getDefaultInstance()))
              .build();
        }
      }
    }
    return getJoinMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static FlowerServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FlowerServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FlowerServiceStub>() {
        @java.lang.Override
        public FlowerServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FlowerServiceStub(channel, callOptions);
        }
      };
    return FlowerServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static FlowerServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FlowerServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FlowerServiceBlockingStub>() {
        @java.lang.Override
        public FlowerServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FlowerServiceBlockingStub(channel, callOptions);
        }
      };
    return FlowerServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static FlowerServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FlowerServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FlowerServiceFutureStub>() {
        @java.lang.Override
        public FlowerServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FlowerServiceFutureStub(channel, callOptions);
        }
      };
    return FlowerServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class FlowerServiceImplBase implements io.grpc.BindableService {

    /**
     */
    public io.grpc.stub.StreamObserver<flwr.android_client.ClientMessage> join(
        io.grpc.stub.StreamObserver<flwr.android_client.ServerMessage> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getJoinMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getJoinMethod(),
            io.grpc.stub.ServerCalls.asyncBidiStreamingCall(
              new MethodHandlers<
                flwr.android_client.ClientMessage,
                flwr.android_client.ServerMessage>(
                  this, METHODID_JOIN)))
          .build();
    }
  }

  /**
   */
  public static final class FlowerServiceStub extends io.grpc.stub.AbstractAsyncStub<FlowerServiceStub> {
    private FlowerServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected FlowerServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FlowerServiceStub(channel, callOptions);
    }

    /**
     */
    public io.grpc.stub.StreamObserver<flwr.android_client.ClientMessage> join(
        io.grpc.stub.StreamObserver<flwr.android_client.ServerMessage> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncBidiStreamingCall(
          getChannel().newCall(getJoinMethod(), getCallOptions()), responseObserver);
    }
  }

  /**
   */
  public static final class FlowerServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<FlowerServiceBlockingStub> {
    private FlowerServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected FlowerServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FlowerServiceBlockingStub(channel, callOptions);
    }
  }

  /**
   */
  public static final class FlowerServiceFutureStub extends io.grpc.stub.AbstractFutureStub<FlowerServiceFutureStub> {
    private FlowerServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected FlowerServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FlowerServiceFutureStub(channel, callOptions);
    }
  }

  private static final int METHODID_JOIN = 0;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final FlowerServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(FlowerServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_JOIN:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.join(
              (io.grpc.stub.StreamObserver<flwr.android_client.ServerMessage>) responseObserver);
        default:
          throw new AssertionError();
      }
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (FlowerServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .addMethod(getJoinMethod())
              .build();
        }
      }
    }
    return result;
  }
}
