#include "OperatorMetadata.hpp"

namespace afd = axis::foundation::definitions;

const afd::OperatorInformation afd::OperatorMetadata::OperatorAnd = 
  afd::OperatorInformation(_T("AND"), 50, kLeftAssociativity, kBinaryOperator);
const afd::OperatorInformation afd::OperatorMetadata::OperatorOr  = 
  afd::OperatorInformation(_T("OR"), 50, kLeftAssociativity, kBinaryOperator);
const afd::OperatorInformation afd::OperatorMetadata::OperatorNot = 
  afd::OperatorInformation(_T("NOT"), 60, kRightAssociativity, kUnaryOperator);
const afd::OperatorInformation afd::OperatorMetadata::OpenGroup   = 
  afd::OperatorInformation(_T("("), 900, kNone, kDelimiter);
const afd::OperatorInformation afd::OperatorMetadata::CloseGroup  = 
  afd::OperatorInformation(_T(")"), 900, kNone, kDelimiter);



afd::OperatorMetadata::OperatorMetadata( void )
{
	// nothing to do here
}

afd::OperatorMetadata::~OperatorMetadata( void )
{
	// nothing to do here
}
