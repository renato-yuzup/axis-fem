#pragma once
#include "NumberFacet.hpp"
#include <xlocale>

namespace axis
{
	namespace services
	{
		namespace locales
		{
			class SystemNumberFacet : public NumberFacet
			{
			private:
				std::locale _systemLocale;

				SystemNumberFacet(const SystemNumberFacet& other);
				SystemNumberFacet& operator =(const SystemNumberFacet& other);

				virtual axis::String DoToPercent( double percent, unsigned int numberOfDigits ) const;

				virtual axis::String DoToString( double number ) const;

				virtual axis::String DoToString( double number, unsigned int numberOfDigits ) const;

				virtual axis::String DoToString( long number ) const;

				virtual axis::String DoToString( unsigned long number ) const;

				virtual axis::String DoToScientificNotation( double number ) const;

				virtual axis::String DoToScientificNotation( double number, int maxDecimalDigits ) const;
			public:
				SystemNumberFacet(void);
				~SystemNumberFacet(void);
			};
		}
	}
}

