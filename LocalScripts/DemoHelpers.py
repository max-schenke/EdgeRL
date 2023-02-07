"""
File:        DemoHelpers.py

Description: This Helper demonstrates how to transfer the XILAPI Python Demos from Python 3.6 to Python 3.9.

Version:     4.0

Date:        May 2021

             dSPACE GmbH shall not be liable for errors contained herein or
             direct, indirect, special, incidental, or consequential damages
             in connection with the furnishing, performance, or use of this
             file.
             Brand names or product names are trademarks or registered
             trademarks of their respective companies or organizations.

Copyright 2021, dSPACE GmbH. All rights reserved.    
"""

from ASAM.XIL.Interfaces.Testbench.Common.ValueContainer import *
from ASAM.XIL.Interfaces.Testbench.Common.ValueContainer.Enum import DataType
from ASAM.XIL.Interfaces.Testbench.Common.Signal.Enum import SegmentTypes
from ASAM.XIL.Interfaces.Testbench.Common.Signal import *

BASE_TYPES = {
DataType.eBOOLEAN: IBooleanValue,
DataType.eINT: IIntValue,
DataType.eUINT: IUintValue,
DataType.eFLOAT: IFloatValue,
DataType.eSTRING: IStringValue,
DataType.eINT_VECTOR: IIntVectorValue,
DataType.eFLOAT_VECTOR: IFloatVectorValue,
DataType.eSTRING_VECTOR: IStringVectorValue,
DataType.eBOOLEAN_VECTOR: IBooleanVectorValue,
DataType.eINT_MATRIX: IIntMatrixValue,
DataType.eFLOAT_MATRIX: IFloatMatrixValue,
DataType.eSTRING_MATRIX: IStringMatrixValue,
DataType.eBOOLEAN_MATRIX: IBooleanMatrixValue,
DataType.eMAP: IMapValue,
DataType.eCURVE: ICurveValue,
DataType.eXYVALUE: IXYValue,
DataType.eUINT_VECTOR: IUintVectorValue,
DataType.eUINT_MATRIX: IUintMatrixValue,
DataType.eSIGNALVALUE: ISignalValue,
DataType.eSIGNALGROUPVALUE: ISignalGroupValue
}

SEGMENT_TYPES = {
SegmentTypes.eOPERATION: IOperationSegment,
SegmentTypes.eSINE: ISineSegment,
SegmentTypes.eEXP: IExpSegment,
SegmentTypes.eNOISE: INoiseSegment,
SegmentTypes.eRAMP: IRampSegment,
SegmentTypes.eRAMPSLOPE: IRampSlopeSegment,
SegmentTypes.eCONST: IConstSegment,
SegmentTypes.eIDLE: IIdleSegment,
SegmentTypes.eSAW: ISawSegment,
SegmentTypes.ePULSE: IPulseSegment,
SegmentTypes.eSIGNALVALUE: ISignalValueSegment,
SegmentTypes.eLOOP: ILoopSegment,
SegmentTypes.eDATAFILE: IDataFileSegment
}

def convertIBaseValue(value):
    """ Convert IBaseValue to special value type """
    valueType = BASE_TYPES.get(value.Type)
    if not valueType:
        raise TypeError(f"Invalid DataType: {value.Type}")
    else:
        return valueType(value)

def convertISignalSegment(segment):
    """ convert the base interface ISignalSegment to special segment type"""
    segmentType = SEGMENT_TYPES.get(segment.Type)
    if not segmentType:
        raise TypeError(f"Invalid SegmentType: {segment.Type}")
    else:
        return segmentType(segment)
