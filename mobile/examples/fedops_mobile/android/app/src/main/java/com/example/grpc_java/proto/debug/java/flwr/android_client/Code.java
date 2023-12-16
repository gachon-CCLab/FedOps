// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: transport.proto

package flwr.android_client;

/**
 * Protobuf enum {@code flwr.proto.Code}
 */
public enum Code
    implements com.google.protobuf.Internal.EnumLite {
  /**
   * <code>OK = 0;</code>
   */
  OK(0),
  /**
   * <code>GET_PROPERTIES_NOT_IMPLEMENTED = 1;</code>
   */
  GET_PROPERTIES_NOT_IMPLEMENTED(1),
  /**
   * <code>GET_PARAMETERS_NOT_IMPLEMENTED = 2;</code>
   */
  GET_PARAMETERS_NOT_IMPLEMENTED(2),
  /**
   * <code>FIT_NOT_IMPLEMENTED = 3;</code>
   */
  FIT_NOT_IMPLEMENTED(3),
  /**
   * <code>EVALUATE_NOT_IMPLEMENTED = 4;</code>
   */
  EVALUATE_NOT_IMPLEMENTED(4),
  UNRECOGNIZED(-1),
  ;

  /**
   * <code>OK = 0;</code>
   */
  public static final int OK_VALUE = 0;
  /**
   * <code>GET_PROPERTIES_NOT_IMPLEMENTED = 1;</code>
   */
  public static final int GET_PROPERTIES_NOT_IMPLEMENTED_VALUE = 1;
  /**
   * <code>GET_PARAMETERS_NOT_IMPLEMENTED = 2;</code>
   */
  public static final int GET_PARAMETERS_NOT_IMPLEMENTED_VALUE = 2;
  /**
   * <code>FIT_NOT_IMPLEMENTED = 3;</code>
   */
  public static final int FIT_NOT_IMPLEMENTED_VALUE = 3;
  /**
   * <code>EVALUATE_NOT_IMPLEMENTED = 4;</code>
   */
  public static final int EVALUATE_NOT_IMPLEMENTED_VALUE = 4;


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
  public static Code valueOf(int value) {
    return forNumber(value);
  }

  public static Code forNumber(int value) {
    switch (value) {
      case 0: return OK;
      case 1: return GET_PROPERTIES_NOT_IMPLEMENTED;
      case 2: return GET_PARAMETERS_NOT_IMPLEMENTED;
      case 3: return FIT_NOT_IMPLEMENTED;
      case 4: return EVALUATE_NOT_IMPLEMENTED;
      default: return null;
    }
  }

  public static com.google.protobuf.Internal.EnumLiteMap<Code>
      internalGetValueMap() {
    return internalValueMap;
  }
  private static final com.google.protobuf.Internal.EnumLiteMap<
      Code> internalValueMap =
        new com.google.protobuf.Internal.EnumLiteMap<Code>() {
          @java.lang.Override
          public Code findValueByNumber(int number) {
            return Code.forNumber(number);
          }
        };

  public static com.google.protobuf.Internal.EnumVerifier 
      internalGetVerifier() {
    return CodeVerifier.INSTANCE;
  }

  private static final class CodeVerifier implements 
       com.google.protobuf.Internal.EnumVerifier { 
          static final com.google.protobuf.Internal.EnumVerifier           INSTANCE = new CodeVerifier();
          @java.lang.Override
          public boolean isInRange(int number) {
            return Code.forNumber(number) != null;
          }
        };

  private final int value;

  private Code(int value) {
    this.value = value;
  }

  // @@protoc_insertion_point(enum_scope:flwr.proto.Code)
}

