#pragma once
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis
{
	namespace domain
	{
		namespace elements
		{
			class AXISCOMMONLIBRARY_API MatrixOption
			{
			public:
				virtual bool DoesRequestStiffnessMatrix(void) const = 0;
				virtual bool DoesRequestConsistentMassMatrix(void) const = 0;
				virtual bool DoesRequestLumpedMassMatrix(void) const = 0;
			};

			class AXISCOMMONLIBRARY_API AllMatricesOption : public MatrixOption
			{
			public:
				virtual bool DoesRequestStiffnessMatrix(void) const;
				virtual bool DoesRequestConsistentMassMatrix(void) const;
				virtual bool DoesRequestLumpedMassMatrix(void) const;
			};

			class AXISCOMMONLIBRARY_API StiffnessMatrixOnlyOption : public MatrixOption
			{
			public:
				virtual bool DoesRequestStiffnessMatrix(void) const;
				virtual bool DoesRequestConsistentMassMatrix(void) const;
				virtual bool DoesRequestLumpedMassMatrix(void) const;
			};

			class AXISCOMMONLIBRARY_API ConsistentMassOnlyOption : public MatrixOption
			{
			public:
				virtual bool DoesRequestStiffnessMatrix(void) const;
				virtual bool DoesRequestConsistentMassMatrix(void) const;
				virtual bool DoesRequestLumpedMassMatrix(void) const;
			};

			class AXISCOMMONLIBRARY_API LumpedMassOnlyOption : public MatrixOption
			{
			public:
				virtual bool DoesRequestStiffnessMatrix(void) const;
				virtual bool DoesRequestConsistentMassMatrix(void) const;
				virtual bool DoesRequestLumpedMassMatrix(void) const;
			};

			class AXISCOMMONLIBRARY_API SomeMatricesOption : public MatrixOption
			{
			private:
				bool _stiffness, _consistent, _lumped;
			public:
				SomeMatricesOption(bool stiffness, bool consistentMass, bool lumpedMass);

				virtual bool DoesRequestStiffnessMatrix(void) const;
				virtual bool DoesRequestConsistentMassMatrix(void) const;
				virtual bool DoesRequestLumpedMassMatrix(void) const;
			};

		}
	}
}
