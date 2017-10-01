/**
 * Autogenerated by Thrift Compiler (0.10.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
package univ.ml.distributed.coreset;

@SuppressWarnings({"cast", "rawtypes", "serial", "unchecked", "unused"})
@javax.annotation.Generated(value = "Autogenerated by Thrift Compiler (0.10.0)", date = "2017-10-01")
public class CoresetPoint implements org.apache.thrift.TBase<CoresetPoint, CoresetPoint._Fields>, java.io.Serializable, Cloneable, Comparable<CoresetPoint> {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("CoresetPoint");

  private static final org.apache.thrift.protocol.TField COORDS_FIELD_DESC = new org.apache.thrift.protocol.TField("coords", org.apache.thrift.protocol.TType.MAP, (short)1);
  private static final org.apache.thrift.protocol.TField DIM_FIELD_DESC = new org.apache.thrift.protocol.TField("dim", org.apache.thrift.protocol.TType.I32, (short)2);

  private static final org.apache.thrift.scheme.SchemeFactory STANDARD_SCHEME_FACTORY = new CoresetPointStandardSchemeFactory();
  private static final org.apache.thrift.scheme.SchemeFactory TUPLE_SCHEME_FACTORY = new CoresetPointTupleSchemeFactory();

  public java.util.Map<java.lang.Integer,java.lang.Double> coords; // required
  public int dim; // required

  /** The set of fields this struct contains, along with convenience methods for finding and manipulating them. */
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    COORDS((short)1, "coords"),
    DIM((short)2, "dim");

    private static final java.util.Map<java.lang.String, _Fields> byName = new java.util.HashMap<java.lang.String, _Fields>();

    static {
      for (_Fields field : java.util.EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }

    /**
     * Find the _Fields constant that matches fieldId, or null if its not found.
     */
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
        case 1: // COORDS
          return COORDS;
        case 2: // DIM
          return DIM;
        default:
          return null;
      }
    }

    /**
     * Find the _Fields constant that matches fieldId, throwing an exception
     * if it is not found.
     */
    public static _Fields findByThriftIdOrThrow(int fieldId) {
      _Fields fields = findByThriftId(fieldId);
      if (fields == null) throw new java.lang.IllegalArgumentException("Field " + fieldId + " doesn't exist!");
      return fields;
    }

    /**
     * Find the _Fields constant that matches name, or null if its not found.
     */
    public static _Fields findByName(java.lang.String name) {
      return byName.get(name);
    }

    private final short _thriftId;
    private final java.lang.String _fieldName;

    _Fields(short thriftId, java.lang.String fieldName) {
      _thriftId = thriftId;
      _fieldName = fieldName;
    }

    public short getThriftFieldId() {
      return _thriftId;
    }

    public java.lang.String getFieldName() {
      return _fieldName;
    }
  }

  // isset id assignments
  private static final int __DIM_ISSET_ID = 0;
  private byte __isset_bitfield = 0;
  public static final java.util.Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    java.util.Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new java.util.EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.COORDS, new org.apache.thrift.meta_data.FieldMetaData("coords", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.MapMetaData(org.apache.thrift.protocol.TType.MAP, 
            new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.I32), 
            new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.DOUBLE))));
    tmpMap.put(_Fields.DIM, new org.apache.thrift.meta_data.FieldMetaData("dim", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.I32)));
    metaDataMap = java.util.Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(CoresetPoint.class, metaDataMap);
  }

  public CoresetPoint() {
  }

  public CoresetPoint(
    java.util.Map<java.lang.Integer,java.lang.Double> coords,
    int dim)
  {
    this();
    this.coords = coords;
    this.dim = dim;
    setDimIsSet(true);
  }

  /**
   * Performs a deep copy on <i>other</i>.
   */
  public CoresetPoint(CoresetPoint other) {
    __isset_bitfield = other.__isset_bitfield;
    if (other.isSetCoords()) {
      java.util.Map<java.lang.Integer,java.lang.Double> __this__coords = new java.util.HashMap<java.lang.Integer,java.lang.Double>(other.coords);
      this.coords = __this__coords;
    }
    this.dim = other.dim;
  }

  public CoresetPoint deepCopy() {
    return new CoresetPoint(this);
  }

  @Override
  public void clear() {
    this.coords = null;
    setDimIsSet(false);
    this.dim = 0;
  }

  public int getCoordsSize() {
    return (this.coords == null) ? 0 : this.coords.size();
  }

  public void putToCoords(int key, double val) {
    if (this.coords == null) {
      this.coords = new java.util.HashMap<java.lang.Integer,java.lang.Double>();
    }
    this.coords.put(key, val);
  }

  public java.util.Map<java.lang.Integer,java.lang.Double> getCoords() {
    return this.coords;
  }

  public CoresetPoint setCoords(java.util.Map<java.lang.Integer,java.lang.Double> coords) {
    this.coords = coords;
    return this;
  }

  public void unsetCoords() {
    this.coords = null;
  }

  /** Returns true if field coords is set (has been assigned a value) and false otherwise */
  public boolean isSetCoords() {
    return this.coords != null;
  }

  public void setCoordsIsSet(boolean value) {
    if (!value) {
      this.coords = null;
    }
  }

  public int getDim() {
    return this.dim;
  }

  public CoresetPoint setDim(int dim) {
    this.dim = dim;
    setDimIsSet(true);
    return this;
  }

  public void unsetDim() {
    __isset_bitfield = org.apache.thrift.EncodingUtils.clearBit(__isset_bitfield, __DIM_ISSET_ID);
  }

  /** Returns true if field dim is set (has been assigned a value) and false otherwise */
  public boolean isSetDim() {
    return org.apache.thrift.EncodingUtils.testBit(__isset_bitfield, __DIM_ISSET_ID);
  }

  public void setDimIsSet(boolean value) {
    __isset_bitfield = org.apache.thrift.EncodingUtils.setBit(__isset_bitfield, __DIM_ISSET_ID, value);
  }

  public void setFieldValue(_Fields field, java.lang.Object value) {
    switch (field) {
    case COORDS:
      if (value == null) {
        unsetCoords();
      } else {
        setCoords((java.util.Map<java.lang.Integer,java.lang.Double>)value);
      }
      break;

    case DIM:
      if (value == null) {
        unsetDim();
      } else {
        setDim((java.lang.Integer)value);
      }
      break;

    }
  }

  public java.lang.Object getFieldValue(_Fields field) {
    switch (field) {
    case COORDS:
      return getCoords();

    case DIM:
      return getDim();

    }
    throw new java.lang.IllegalStateException();
  }

  /** Returns true if field corresponding to fieldID is set (has been assigned a value) and false otherwise */
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new java.lang.IllegalArgumentException();
    }

    switch (field) {
    case COORDS:
      return isSetCoords();
    case DIM:
      return isSetDim();
    }
    throw new java.lang.IllegalStateException();
  }

  @Override
  public boolean equals(java.lang.Object that) {
    if (that == null)
      return false;
    if (that instanceof CoresetPoint)
      return this.equals((CoresetPoint)that);
    return false;
  }

  public boolean equals(CoresetPoint that) {
    if (that == null)
      return false;
    if (this == that)
      return true;

    boolean this_present_coords = true && this.isSetCoords();
    boolean that_present_coords = true && that.isSetCoords();
    if (this_present_coords || that_present_coords) {
      if (!(this_present_coords && that_present_coords))
        return false;
      if (!this.coords.equals(that.coords))
        return false;
    }

    boolean this_present_dim = true;
    boolean that_present_dim = true;
    if (this_present_dim || that_present_dim) {
      if (!(this_present_dim && that_present_dim))
        return false;
      if (this.dim != that.dim)
        return false;
    }

    return true;
  }

  @Override
  public int hashCode() {
    int hashCode = 1;

    hashCode = hashCode * 8191 + ((isSetCoords()) ? 131071 : 524287);
    if (isSetCoords())
      hashCode = hashCode * 8191 + coords.hashCode();

    hashCode = hashCode * 8191 + dim;

    return hashCode;
  }

  @Override
  public int compareTo(CoresetPoint other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }

    int lastComparison = 0;

    lastComparison = java.lang.Boolean.valueOf(isSetCoords()).compareTo(other.isSetCoords());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetCoords()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.coords, other.coords);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = java.lang.Boolean.valueOf(isSetDim()).compareTo(other.isSetDim());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetDim()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.dim, other.dim);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    return 0;
  }

  public _Fields fieldForId(int fieldId) {
    return _Fields.findByThriftId(fieldId);
  }

  public void read(org.apache.thrift.protocol.TProtocol iprot) throws org.apache.thrift.TException {
    scheme(iprot).read(iprot, this);
  }

  public void write(org.apache.thrift.protocol.TProtocol oprot) throws org.apache.thrift.TException {
    scheme(oprot).write(oprot, this);
  }

  @Override
  public java.lang.String toString() {
    java.lang.StringBuilder sb = new java.lang.StringBuilder("CoresetPoint(");
    boolean first = true;

    sb.append("coords:");
    if (this.coords == null) {
      sb.append("null");
    } else {
      sb.append(this.coords);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("dim:");
    sb.append(this.dim);
    first = false;
    sb.append(")");
    return sb.toString();
  }

  public void validate() throws org.apache.thrift.TException {
    // check for required fields
    // check for sub-struct validity
  }

  private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
    try {
      write(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(out)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }

  private void readObject(java.io.ObjectInputStream in) throws java.io.IOException, java.lang.ClassNotFoundException {
    try {
      // it doesn't seem like you should have to do this, but java serialization is wacky, and doesn't call the default constructor.
      __isset_bitfield = 0;
      read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }

  private static class CoresetPointStandardSchemeFactory implements org.apache.thrift.scheme.SchemeFactory {
    public CoresetPointStandardScheme getScheme() {
      return new CoresetPointStandardScheme();
    }
  }

  private static class CoresetPointStandardScheme extends org.apache.thrift.scheme.StandardScheme<CoresetPoint> {

    public void read(org.apache.thrift.protocol.TProtocol iprot, CoresetPoint struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TField schemeField;
      iprot.readStructBegin();
      while (true)
      {
        schemeField = iprot.readFieldBegin();
        if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
          break;
        }
        switch (schemeField.id) {
          case 1: // COORDS
            if (schemeField.type == org.apache.thrift.protocol.TType.MAP) {
              {
                org.apache.thrift.protocol.TMap _map0 = iprot.readMapBegin();
                struct.coords = new java.util.HashMap<java.lang.Integer,java.lang.Double>(2*_map0.size);
                int _key1;
                double _val2;
                for (int _i3 = 0; _i3 < _map0.size; ++_i3)
                {
                  _key1 = iprot.readI32();
                  _val2 = iprot.readDouble();
                  struct.coords.put(_key1, _val2);
                }
                iprot.readMapEnd();
              }
              struct.setCoordsIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 2: // DIM
            if (schemeField.type == org.apache.thrift.protocol.TType.I32) {
              struct.dim = iprot.readI32();
              struct.setDimIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          default:
            org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
        }
        iprot.readFieldEnd();
      }
      iprot.readStructEnd();

      // check for required fields of primitive type, which can't be checked in the validate method
      struct.validate();
    }

    public void write(org.apache.thrift.protocol.TProtocol oprot, CoresetPoint struct) throws org.apache.thrift.TException {
      struct.validate();

      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.coords != null) {
        oprot.writeFieldBegin(COORDS_FIELD_DESC);
        {
          oprot.writeMapBegin(new org.apache.thrift.protocol.TMap(org.apache.thrift.protocol.TType.I32, org.apache.thrift.protocol.TType.DOUBLE, struct.coords.size()));
          for (java.util.Map.Entry<java.lang.Integer, java.lang.Double> _iter4 : struct.coords.entrySet())
          {
            oprot.writeI32(_iter4.getKey());
            oprot.writeDouble(_iter4.getValue());
          }
          oprot.writeMapEnd();
        }
        oprot.writeFieldEnd();
      }
      oprot.writeFieldBegin(DIM_FIELD_DESC);
      oprot.writeI32(struct.dim);
      oprot.writeFieldEnd();
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }

  }

  private static class CoresetPointTupleSchemeFactory implements org.apache.thrift.scheme.SchemeFactory {
    public CoresetPointTupleScheme getScheme() {
      return new CoresetPointTupleScheme();
    }
  }

  private static class CoresetPointTupleScheme extends org.apache.thrift.scheme.TupleScheme<CoresetPoint> {

    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, CoresetPoint struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TTupleProtocol oprot = (org.apache.thrift.protocol.TTupleProtocol) prot;
      java.util.BitSet optionals = new java.util.BitSet();
      if (struct.isSetCoords()) {
        optionals.set(0);
      }
      if (struct.isSetDim()) {
        optionals.set(1);
      }
      oprot.writeBitSet(optionals, 2);
      if (struct.isSetCoords()) {
        {
          oprot.writeI32(struct.coords.size());
          for (java.util.Map.Entry<java.lang.Integer, java.lang.Double> _iter5 : struct.coords.entrySet())
          {
            oprot.writeI32(_iter5.getKey());
            oprot.writeDouble(_iter5.getValue());
          }
        }
      }
      if (struct.isSetDim()) {
        oprot.writeI32(struct.dim);
      }
    }

    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, CoresetPoint struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TTupleProtocol iprot = (org.apache.thrift.protocol.TTupleProtocol) prot;
      java.util.BitSet incoming = iprot.readBitSet(2);
      if (incoming.get(0)) {
        {
          org.apache.thrift.protocol.TMap _map6 = new org.apache.thrift.protocol.TMap(org.apache.thrift.protocol.TType.I32, org.apache.thrift.protocol.TType.DOUBLE, iprot.readI32());
          struct.coords = new java.util.HashMap<java.lang.Integer,java.lang.Double>(2*_map6.size);
          int _key7;
          double _val8;
          for (int _i9 = 0; _i9 < _map6.size; ++_i9)
          {
            _key7 = iprot.readI32();
            _val8 = iprot.readDouble();
            struct.coords.put(_key7, _val8);
          }
        }
        struct.setCoordsIsSet(true);
      }
      if (incoming.get(1)) {
        struct.dim = iprot.readI32();
        struct.setDimIsSet(true);
      }
    }
  }

  private static <S extends org.apache.thrift.scheme.IScheme> S scheme(org.apache.thrift.protocol.TProtocol proto) {
    return (org.apache.thrift.scheme.StandardScheme.class.equals(proto.getScheme()) ? STANDARD_SCHEME_FACTORY : TUPLE_SCHEME_FACTORY).getScheme();
  }
}

