#ifndef __SECTIONDEFINITION_HPP
#define __SECTIONDEFINITION_HPP
#include "foundation/Axis.Physalis.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "AxisString.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "domain/collections/FlagCollection.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class AXISPHYSALIS_API SectionDefinition
{
public:
	SectionDefinition(const axis::String& name, 
    const axis::services::language::syntax::evaluation::ParameterValue& elemDescription);
	SectionDefinition(const axis::String& name);
	SectionDefinition(const SectionDefinition& definition);
	virtual ~SectionDefinition(void);

	axis::String GetSectionTypeName(void) const;

	void AddProperty(const axis::String& name, 
    axis::services::language::syntax::evaluation::ParameterValue& value);
	void AddFlag(const axis::String& flag);

	axis::services::language::syntax::evaluation::ParameterValue& GetPropertyValue(
    const axis::String& propertyName) const;
	bool IsPropertyDefined(const axis::String& propertyName) const;
	bool IsFlagSet(const axis::String& flag) const;
	unsigned int PropertyCount(void) const;
	unsigned int FlagCount(void) const;
				
	const axis::domain::materials::MaterialModel& GetMaterial(void) const;
	void SetMaterial(const axis::domain::materials::MaterialModel& material);

	SectionDefinition& operator =(const SectionDefinition& other);

	bool operator ==(const SectionDefinition& other) const;
	bool operator !=(const SectionDefinition& other) const;

	SectionDefinition& Clone(void) const;
private:
	void Copy(const SectionDefinition& definition);
	axis::services::language::syntax::evaluation::ParameterList &ParseParameters(
    const axis::services::language::syntax::evaluation::ParameterValue& params) const;

	axis::String _sectionTypeName;
	axis::services::language::syntax::evaluation::ParameterList *_paramList;
	axis::domain::collections::FlagCollection *_flagList;
	const axis::domain::materials::MaterialModel *_material;
};

} } } } // namespace axis::application::parsing::core
#endif
