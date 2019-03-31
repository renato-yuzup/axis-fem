#pragma once
#include "VariableConstraintFactory.hpp"

namespace axis { namespace application { namespace factories { namespace boundary_conditions {

class PrescribedDisplacementConstraintFactory : public VariableConstraintFactory
{
public:
	PrescribedDisplacementConstraintFactory(void);
protected:
  virtual void RegisterNewBoundaryCondition( axis::application::jobs::AnalysisStep& step, 
    axis::domain::elements::DoF& dof, axis::foundation::memory::RelativePointer& curvePtr, 
    real scaleFactor, real releaseTime );
};

} } } } // namespace axis::application::factories::boundary_conditions
