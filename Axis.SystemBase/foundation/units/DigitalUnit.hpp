#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis
{
	namespace foundation
	{
		namespace units
		{
			class AXISSYSTEMBASE_API DigitalUnit
			{
			private:
				DigitalUnit(void);
			public:
				class AXISSYSTEMBASE_API MultipleUnit
				{
				public:
					virtual uint64 GetScaleFactor(void) const = 0;
				};
				class AXISSYSTEMBASE_API ByteUnit : public MultipleUnit
				{
				public:
					virtual uint64 GetScaleFactor(void) const;
				};
				class AXISSYSTEMBASE_API KiloByteUnit : public MultipleUnit
				{
				public:
					virtual uint64 GetScaleFactor(void) const;
				};
				class AXISSYSTEMBASE_API MegaByteUnit : public MultipleUnit
				{
				public:
					virtual uint64 GetScaleFactor(void) const;
				};
				class AXISSYSTEMBASE_API GigaByteUnit : public MultipleUnit
				{
				public:
					virtual uint64 GetScaleFactor(void) const;
				};
				class AXISSYSTEMBASE_API TeraByteUnit : public MultipleUnit
				{
				public:
					virtual uint64 GetScaleFactor(void) const;
				};

				class AXISSYSTEMBASE_API FromScale
				{
				private:
					uint64 _scaleValue;
				public:
					FromScale(const MultipleUnit& unit);
					uint64 Convert(uint64 valueInScale) const;
				};

				class AXISSYSTEMBASE_API ToScale
				{
				private:
					uint64 _scaleValue;
				public:
					ToScale(const MultipleUnit& unit);
					uint64 Convert(uint64 valueInBytes) const;
				};

				~DigitalUnit(void);

				static uint64 Convert(uint64 value, const FromScale& from, const ToScale& to);
				static double Convert(uint64 value, const FromScale& from, const ToScale& to, int maxDecimalDigits);
			};
		}
	}
}

