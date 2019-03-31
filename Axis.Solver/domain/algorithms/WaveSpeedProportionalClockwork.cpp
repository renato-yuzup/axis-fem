#include "WaveSpeedProportionalClockwork.hpp"
#include <omp.h>
#include "domain/elements/ElementGeometry.hpp"

namespace adal = axis::domain::algorithms;
namespace ada  = axis::domain::analyses;
namespace adc  = axis::domain::collections;
namespace ade  = axis::domain::elements;
namespace afb  = axis::foundation::blas;

const int chunkSize = 20;

adal::WaveSpeedProportionalClockwork::WaveSpeedProportionalClockwork( 
  real dtimeScaleFactor, bool considerNonLinearity )
{
	_timeIncrementScaleFactor = dtimeScaleFactor;
	_considerNonLinearity = considerNonLinearity;
	_lastTimeIncrement = -1;
}

adal::WaveSpeedProportionalClockwork::~WaveSpeedProportionalClockwork( void )
{
	// nothing to do here
}

void adal::WaveSpeedProportionalClockwork::CalculateNextTick( 
  ada::AnalysisTimeline& ti, ada::NumericalModel& model )
{
	CalculateNextTick(ti, model, -1);
}

void adal::WaveSpeedProportionalClockwork::CalculateNextTick( 
  ada::AnalysisTimeline& ti, ada::NumericalModel& model, real maxTimeIncrement )
{
	// if we are not considering nonlinearity, calculating only by the first time
	// the time increment will suffice
	if (!_considerNonLinearity && _lastTimeIncrement > 0)
	{
		ti.NextTimeIncrement() = _lastTimeIncrement > maxTimeIncrement ? 
      maxTimeIncrement : _lastTimeIncrement;
		return;
	}

	// we're going to iterate through each element in the model and get the 
	// smaller timestep found
	adc::ElementSet& elements = model.Elements();
	size_type count = elements.Count();
	
	// initializations
	int workerCount = omp_get_num_threads();
	
	real *localMinima = new real[workerCount];
	for (int i = 0; i < workerCount; ++i)
	{
		localMinima[i] = -1;
	}

	size_type actualChunkSize = ((count / workerCount) < chunkSize? 
    (count / workerCount) : chunkSize);
	if (actualChunkSize == 0) ++actualChunkSize;
	size_type nextChunkIndex = 0;

	#pragma omp parallel for num_threads(workerCount)
	for (int i = 0; i < workerCount; i++)
	{
		int workerIdx = omp_get_thread_num();
		size_type myChunkStartIdx;
		do 
		{
			myChunkStartIdx = count;
			#pragma omp critical (__wavespeed_clockwork_critical_region__)
			{
				if (nextChunkIndex < count)
				{
					myChunkStartIdx = nextChunkIndex;
					nextChunkIndex += actualChunkSize;
				}
			}

			// when we entered the loop, another thread might have got the last chunk, 
			// so we a double check is important
			if (myChunkStartIdx != count)
			{
				for (size_type idx = 0; idx < actualChunkSize; ++idx)
				{
					ade::FiniteElement& element = 
            elements.GetByInternalIndex(myChunkStartIdx + idx);
					const afb::ColumnVector& globalDispl = model.Kinematics().Displacement();
					afb::ColumnVector localDisplacement(element.Geometry().GetTotalDofCount());
					element.ExtractLocalField(localDisplacement, globalDispl);
					real waveSpeed = MATH_ABS(element.GetCriticalTimestep(localDisplacement));
          if (waveSpeed <= 0)
          { // TODO: exclude when code is reliable enough!
            throw;
          }
					// compare with previous results
					if ((localMinima[workerIdx] < 0 && waveSpeed > 0) || // first comparison
						 (localMinima[workerIdx] > 0 && waveSpeed < localMinima[workerIdx] 
             && waveSpeed > 0))	// a new minimal was found
					{
						localMinima[workerIdx] = waveSpeed;
					}
				}
			}
		} while (myChunkStartIdx != count);
	}

	// check for the smaller wave speed in all worker threads results
	real nextTimeIncrement = localMinima[0];
	for (int i = 1; i < workerCount; ++i)
	{
		if (localMinima[i] < nextTimeIncrement && localMinima[i] > 0)
		{
			nextTimeIncrement = localMinima[i];
		}
	}

  if (nextTimeIncrement < 0)
  { // TODO: exclude when code is reliable enough!
    throw;
  }

	// free resources
	delete [] localMinima;

  nextTimeIncrement *= _timeIncrementScaleFactor;
	_lastTimeIncrement = ((nextTimeIncrement > maxTimeIncrement) && 
    (maxTimeIncrement > 0))? maxTimeIncrement : nextTimeIncrement;
	ti.NextTimeIncrement() = _lastTimeIncrement;
}

void adal::WaveSpeedProportionalClockwork::Destroy( void ) const
{
	delete this;
}

