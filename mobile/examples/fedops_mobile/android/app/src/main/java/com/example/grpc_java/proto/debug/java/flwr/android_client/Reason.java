// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: transport.proto

package flwr.android_client;

/**
 * Protobuf enum {@code flwr.proto.Reason}
 */
public enum Reason
    implements com.google.protobuf.Internal.EnumLite {
  /**
   * <code>UNKNOWN = 0;</code>
   */
  UNKNOWN(0),
  /**
   * <code>RECONNECT = 1;</code>
   */
  RECONNECT(1),
  /**
   * <code>POWER_DISCONNECTED = 2;</code>
   */
  POWER_DISCONNECTED(2),
  /**
   * <code>WIFI_UNAVAILABLE = 3;</code>
   */
  WIFI_UNAVAILABLE(3),
  /**
   * <code>ACK = 4;</code>
   */
  ACK(4),
  UNRECOGNIZED(-1),
  ;

  /**
   * <code>UNKNOWN = 0;</code>
   */
  public static final int UNKNOWN_VALUE = 0;
  /**
   * <code>RECONNECT = 1;</code>
   */
  public static final int RECONNECT_VALUE = 1;
  /**
   * <code>POWER_DISCONNECTED = 2;</code>
   */
  public static final int POWER_DISCONNECTED_VALUE = 2;
  /**
   * <code>WIFI_UNAVAILABLE = 3;</code>
   */
  public static final int WIFI_UNAVAILABLE_VALUE = 3;
  /**
   * <code>ACK = 4;</code>
   */
  public static final int ACK_VALUE = 4;


  @java.lang.Override
  public final int getNumber() {
    if (this == UNRECOGNIZED) {
      throw new java.lang.IllegalArgumentException(
          "Can't get the number of an unknown enum value.");
    }
    return value;
  }

  /**
   * @param value The number of the enum to look for.
   * @return The enum associated with the given number.
   * @deprecated Use {@link #forNumber(int)} instead.
   */
  @java.lang.Deprecated
  public static Reason valueOf(int value) {
    return forNumber(value);
  }

  public static Reason forNumber(int value) {
    switch (value) {
      case 0: return UNKNOWN;
      case 1: return RECONNECT;
      case 2: return POWER_DISCONNECTED;
      case 3: return WIFI_UNAVAILABLE;
      case 4: return ACK;
      default: return null;
    }
  }

  public static com.google.protobuf.Internal.EnumLiteMap<Reason>
      internalGetValueMap() {
    return internalValueMap;
  }
  private static final com.google.protobuf.Internal.EnumLiteMap<
      Reason> internalValueMap =
        new com.google.protobuf.Internal.EnumLiteMap<Reason>() {
          @java.lang.Override
          public Reason findValueByNumber(int number) {
            return Reason.forNumber(number);
          }
        };

  public static com.google.protobuf.Internal.EnumVerifier 
      internalGetVerifier() {
    return ReasonVerifier.INSTANCE;
  }

  private static final class ReasonVerifier implements 
       com.google.protobuf.Internal.EnumVerifier { 
          static final com.google.protobuf.Internal.EnumVerifier           INSTANCE = new ReasonVerifier();
          @java.lang.Override
          public boolean isInRange(int number) {
            return Reason.forNumber(number) != null;
          }
        };

  private final int value;

  private Reason(int value) {
    this.value = value;
  }

  // @@protoc_insertion_point(enum_scope:flwr.proto.Reason)
}

