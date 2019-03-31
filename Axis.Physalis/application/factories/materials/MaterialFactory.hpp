#pragma once
#include "AxisString.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "foundation/Axis.Physalis.hpp"

namespace axis { namespace application { namespace factories { namespace materials {

class AXISPHYSALIS_API MaterialFactory
{
public:
	virtual ~MaterialFactory(void);
	virtual void Destroy(void) const = 0;
	virtual bool CanBuild(const axis::String& modelName, 
    const axis::services::language::syntax::evaluation::ParameterList& params) const = 0;
	virtual axis::domain::materials::MaterialModel& Build(const axis::String& modelName, 
    const axis::services::language::syntax::evaluation::ParameterList& params) = 0;
};			

} } } } // namespace axis::application::factories::materials
