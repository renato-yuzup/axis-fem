#pragma once
#include "domain/algorithms/Clockwork.hpp"

namespace axis
{
	namespace domain
	{
		namespace algorithms
		{
			class WaveSpeedProportionalClockwork : public Clockwork
			{
			private:
				bool _considerNonLinearity;
				real _lastTimeIncrement;
				real _timeIncrementScaleFactor;

				real DivideAndConquerCalculateSmallerPropagationSpeed(axis::domain::collections::ElementSet& elements, size_type fromIndex, size_type toIndex);
			public:
				WaveSpeedProportionalClockwork(real dtimeScaleFactor, bool considerNonLinearity);
				~WaveSpeedProportionalClockwork(void);

				virtual void CalculateNextTick( axis::domain::analyses::AnalysisTimeline& ti, axis::domain::analyses::NumericalModel& model );

				virtual void CalculateNextTick( axis::domain::analyses::AnalysisTimeline& ti, axis::domain::analyses::NumericalModel& model, real maxTimeIncrement );

				virtual void Destroy( void ) const;

			};
		}
	}
}

