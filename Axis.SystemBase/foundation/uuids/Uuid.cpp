#include "Uuid.hpp"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "foundation/OutOfBoundsException.hpp"
#include <boost/functional/hash/extensions.hpp>


const int axis::foundation::uuids::Uuid::Length = 16;

axis::foundation::uuids::Uuid::Uuid( void )
{
	for (int i = 0; i < Length; ++i)
	{
		_bytes[i] = 0x00;
	}
}

axis::foundation::uuids::Uuid::Uuid( const byte * const uuid )
{
	for (int i = 0; i < Length; ++i)
	{
		_bytes[i] = uuid[i];
	}
}

axis::foundation::uuids::Uuid::Uuid( const int * const uuid )
{
	for (int i = 0; i < Length; ++i)
	{
		_bytes[i] = (byte)uuid[i];
	}
}

axis::foundation::uuids::Uuid::Uuid( const unsigned int * const uuid )
{
	for (int i = 0; i < Length; ++i)
	{
		_bytes[i] = (byte)uuid[i];
	}
}

axis::foundation::uuids::Uuid::Uuid( const Uuid& uuid )
{
	for (int i = 0; i < Length; ++i)
	{
		_bytes[i] = uuid._bytes[i];
	}
}

axis::foundation::uuids::Uuid::~Uuid( void )
{
	// nothing to do here
}

axis::String axis::foundation::uuids::Uuid::ToString( void ) const
{
	return _T("{") + ToStringUnbraced() + _T("}");
}

axis::String axis::foundation::uuids::Uuid::ToStringUnbraced( void ) const
{
	axis::String s; s.reserve(40);
	for (int i = 0; i < 4; ++i)
	{
		s.append(axis::String::int_to_hex((long)_bytes[i], 2).replace(_T(" "), _T("0")));
	}
	for (int i = 4; i < 10; ++i)
	{
		if (i % 2 == 0)
		{
			s.append(_T("-"));
		}
		s.append(axis::String::int_to_hex((long)_bytes[i], 2).replace(_T(" "), _T("0")));
	}
	s.append(_T("-"));
	for (int i = 10; i < 16; ++i)
	{
		s.append(axis::String::int_to_hex((long)_bytes[i], 2).replace(_T(" "), _T("0")));
	}
	return s;
}

axis::String axis::foundation::uuids::Uuid::ToStringAsByteArray( void ) const
{
	axis::String s; s.reserve(50);
	for (int i = 0; i < 16; ++i)
	{
		s.append(axis::String::int_to_hex((long)_bytes[i], 2).replace(_T(" "), _T("0")));
		if (i < 15)
		{
			s.append(_T("-"));
		}
	}
	return s;
}

axis::String axis::foundation::uuids::Uuid::ToStringAsByteSequence( void ) const
{
	axis::String s; s.reserve(50);
	for (int i = 0; i < 16; ++i)
	{
		s.append(axis::String::int_to_hex((long)_bytes[i], 2).replace(_T(" "), _T("0")));
	}
	return s;
}

bool axis::foundation::uuids::Uuid::operator==( const Uuid& uuid ) const
{
	for (int i = 0; i < Length; ++i)
	{
		if (_bytes[i] != uuid._bytes[i])
		{
			return false;
		}
	}
	return true;
}

bool axis::foundation::uuids::Uuid::operator!=( const Uuid& uuid ) const
{
	return !(*this == uuid);
}

axis::foundation::uuids::Uuid::byte axis::foundation::uuids::Uuid::operator[]( int index ) const
{
	if (index < 0 || index > Length)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	return _bytes[index];
}

axis::foundation::uuids::Uuid::byte axis::foundation::uuids::Uuid::GetByte( int index ) const
{
	if (index < 0 || index > Length)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	return _bytes[index];
}

axis::foundation::uuids::Uuid& axis::foundation::uuids::Uuid::operator=( const Uuid& uuid )
{
	for (int i = 0; i < Length; ++i)
	{
		_bytes[i] = uuid._bytes[i];
	}
	return *this;
}

axis::foundation::uuids::Uuid axis::foundation::uuids::Uuid::GenerateRandom( void )
{
	boost::uuids::uuid uuid = boost::uuids::random_generator()();
	return Uuid(uuid.data);
}

bool axis::foundation::uuids::Uuid::operator<( const Uuid& uuid ) const
{
	for (int i = 0; i < Length; ++i)
	{
		if (_bytes[i] < uuid._bytes[i]) return true;
	}
	return false;
}

bool axis::foundation::uuids::Uuid::operator>( const Uuid& uuid ) const
{
	for (int i = 0; i < Length; ++i)
	{
		if (_bytes[i] > uuid._bytes[i]) return true;
	}
	return false;
}

bool axis::foundation::uuids::Uuid::operator<=( const Uuid& uuid ) const
{
	for (int i = 0; i < Length; ++i)
	{
		if (_bytes[i] <= uuid._bytes[i]) return true;
	}
	return false;
}

bool axis::foundation::uuids::Uuid::operator>=( const Uuid& uuid ) const
{
	for (int i = 0; i < Length; ++i)
	{
		if (_bytes[i] >= uuid._bytes[i]) return true;
	}
	return false;
}

size_t axis::foundation::uuids::hash_value( const Uuid& uuid )
{
  boost::hash<boost::uuids::uuid> uuid_hasher;
  boost::uuids::string_generator gen;
  boost::uuids::uuid uuidVal = gen(uuid.ToStringAsByteSequence());
  return uuid_hasher(uuidVal);
}
