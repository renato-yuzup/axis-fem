#pragma once
#include "CollationFacet.hpp"
#include <xlocale>

namespace axis
{
	namespace services
	{
		namespace locales
		{
			class SystemCollationFacet : public CollationFacet
			{
			private:
				std::locale _systemLocale;

				SystemCollationFacet(const SystemCollationFacet& other);
				SystemCollationFacet& operator =(const SystemCollationFacet& other);

				virtual int DoCompareStrict( const axis::String& s1, const axis::String& s2 ) const;

				virtual int DoCompareIgnoreCase( const axis::String& s1, const axis::String& s2 ) const;

				virtual int DoCompareIgnoreCaseAndAccents( const axis::String& s1, const axis::String& s2 ) const;

				virtual DefaultCollationType GetDefaultCollation( void ) const;
			public:
				SystemCollationFacet(void);
			};
		}
	}
}

