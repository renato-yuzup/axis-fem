#include "DigitalUnit.hpp"
#include <math.h>
#include "foundation/ArgumentException.hpp"

static const uint64 UnitMultiple = 1000;

axis::foundation::units::DigitalUnit::DigitalUnit( void )
{
	// nothing to do here
}

axis::foundation::units::DigitalUnit::~DigitalUnit( void )
{
	// nothing to do here
}

uint64 axis::foundation::units::DigitalUnit::Convert( uint64 value, const FromScale& from, const ToScale& to )
{
	uint64 valueInBytes = from.Convert(value);
	uint64 valueInNewScale = to.Convert(valueInBytes);
	return valueInNewScale;
}

double axis::foundation::units::DigitalUnit::Convert( uint64 value, const FromScale& from, const ToScale& to, int maxDecimalDigits )
{
	if (maxDecimalDigits < 0)
	{
		throw axis::foundation::ArgumentException(_T("Invalid decimal digits count."));
	}
	uint64 valueInBytes = from.Convert(value) * (uint64)::pow(10.0, maxDecimalDigits);
	uint64 valueInNewScale = to.Convert(valueInBytes);
	
	double realValue = valueInNewScale / ::pow(10.0, (double)maxDecimalDigits);

	return realValue;
}

uint64 axis::foundation::units::DigitalUnit::ByteUnit::GetScaleFactor( void ) const
{
	return 1;
}

uint64 axis::foundation::units::DigitalUnit::KiloByteUnit::GetScaleFactor( void ) const
{
	return UnitMultiple;
}

uint64 axis::foundation::units::DigitalUnit::MegaByteUnit::GetScaleFactor( void ) const
{
	return UnitMultiple*UnitMultiple;
}

uint64 axis::foundation::units::DigitalUnit::GigaByteUnit::GetScaleFactor( void ) const
{
	return UnitMultiple*UnitMultiple*UnitMultiple;
}

uint64 axis::foundation::units::DigitalUnit::TeraByteUnit::GetScaleFactor( void ) const
{
	return UnitMultiple*UnitMultiple*UnitMultiple*UnitMultiple;
}

axis::foundation::units::DigitalUnit::FromScale::FromScale( const MultipleUnit& unit )
{
	_scaleValue = unit.GetScaleFactor();
}

uint64 axis::foundation::units::DigitalUnit::FromScale::Convert( uint64 valueInScale ) const
{
	return valueInScale * _scaleValue;
}

axis::foundation::units::DigitalUnit::ToScale::ToScale( const MultipleUnit& unit )
{
	_scaleValue = unit.GetScaleFactor();
}

uint64 axis::foundation::units::DigitalUnit::ToScale::Convert( uint64 valueInBytes ) const
{
	return valueInBytes / _scaleValue;
}

