// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: transport.proto

package flwr.android_client;

public interface StatusOrBuilder extends
    // @@protoc_insertion_point(interface_extends:flwr.proto.Status)
    com.google.protobuf.MessageLiteOrBuilder {

  /**
   * <code>.flwr.proto.Code code = 1;</code>
   * @return The enum numeric value on the wire for code.
   */
  int getCodeValue();
  /**
   * <code>.flwr.proto.Code code = 1;</code>
   * @return The code.
   */
  flwr.android_client.Code getCode();

  /**
   * <code>string message = 2;</code>
   * @return The message.
   */
  java.lang.String getMessage();
  /**
   * <code>string message = 2;</code>
   * @return The bytes for message.
   */
  com.google.protobuf.ByteString
      getMessageBytes();
}
