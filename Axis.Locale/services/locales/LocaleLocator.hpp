#pragma once
#include "AxisString.hpp"
#include "foundation/Axis.Locale.hpp"
#include "services/locales/Locale.hpp"

namespace axis
{
	namespace services
	{
		namespace locales
		{
			/**
			 * Provides locale configuration and lookup services.
			 */
			class AXISLOCALE_API LocaleLocator
			{
			private:
				class LocaleData;
				class LocaleDescriptor;

				
				static LocaleLocator *_singleInstance;

				LocaleData *_localeData;
				LocaleDescriptor *_localeComponents;


				LocaleLocator(void);

				// disallow copy construction or copy assignment
				LocaleLocator(const LocaleLocator& other);
				LocaleLocator& operator =(const LocaleLocator& other);
			public:

				/**
				 * Destructor.
				 */
				~LocaleLocator(void);

				/**
				 * Returns a reference to the single instance locale locator.
				 *
				 * @return The locator.
				 */
				static LocaleLocator& GetLocator(void);

				/**
				 * Returns the default locale used by the application.
				 *
				 * @return The default locale.
				 */
				const Locale& GetDefaultLocale(void) const;

				/**
				 * Returns the locale that matches with the current
				 * regional configuration of the operating system.
				 *
				 * @return The system locale.
				 */
				const Locale& GetSystemRegionLocale(void) const;

				/**
				 * Returns the locale used by parts of the application that support localization.
				 *
				 * @return The current in use locale.
				 */
				const Locale& GetGlobalLocale(void) const;

				/**
				 * Sets the new locale to used by parts of the application that support localization.
				 *
				 * @param locale The new locale to use.
				 */
				void SetGlobalLocale(const Locale& locale);

				/**
				 * Returns a known locale associated to a locale code as
				 * defined in the POSIX locale naming convention.
				 *
				 * @param localeCode The POSIX locale code.
				 *
				 * @return The locale.
				 */
				const Locale& GetLocale(const axis::String& localeCode) const;

				/**
				 * Loads additional locale definitions from the application directory.
				 */
				void LoadLocales(void);

				/**
				 * Unloads all locales externally defined.
				 */
				void UnloadLocales(void);

				/**
				 * Returns the number of registered locales.
				 *
				 * @return The registered locale count.
				 */
				size_type GetRegisteredLocaleCount(void) const;

				/**
				 * Queries if there is a known locale associated to a POSIX 
				 * locale code.
				 *
				 * @param localeCode The POSIX locale code.
				 *
				 * @return true if registered, false if not.
				 */
				bool IsRegistered(const axis::String& localeCode) const;
			};
		}
	}
}

