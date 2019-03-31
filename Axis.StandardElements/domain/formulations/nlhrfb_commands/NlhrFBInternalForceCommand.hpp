#pragma once
#include "domain/formulations/InternalForceCommand.hpp"

namespace axis { namespace domain { namespace formulations { 
namespace nlhrfb_commands {

	class NlhrFBInternalForceCommand : 
		public axis::domain::formulations::InternalForceCommand
	{
	public:
		NlhrFBInternalForceCommand(void);
		~NlhrFBInternalForceCommand(void);
	private:
		virtual void DoRun( uint64 numThreadsToUse, uint64 startIndex, 
			void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
			const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize, 
			axis::foundation::memory::RelativePointer& reducedModelPtr, 
			real currentTime, real lastTimeIncrement, real nextTimeIncrement );
	};

} } } } // namespace axis::domain::formulations::nlhr_commands