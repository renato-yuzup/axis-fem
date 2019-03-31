#include "SectionDefinition.hpp"
#include "services/language/syntax/evaluation/ArrayValue.hpp"
#include "services/language/syntax/ParameterListParser.hpp"
#include "services/language/parsing/EnumerationExpression.hpp"
#include "services/language/syntax/evaluation/ParameterAssignment.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "foundation/ArgumentException.hpp"

namespace aapc = axis::application::parsing::core;
namespace adm = axis::domain::materials;
namespace asls = axis::services::language::syntax;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aslp = axis::services::language::parsing;

aapc::SectionDefinition::SectionDefinition( const axis::String& name, 
                                            const aslse::ParameterValue& elemDescription)
{
	_sectionTypeName = name;
	_flagList = new axis::domain::collections::FlagCollection();
	_material = NULL;
	_paramList = &ParseParameters(elemDescription);	
}

aapc::SectionDefinition::SectionDefinition( const SectionDefinition& definition )
{
	_paramList = NULL; _flagList = NULL; _material = NULL;
	Copy(definition);
}

aapc::SectionDefinition::SectionDefinition( const axis::String& name )
{
	_sectionTypeName = name;
	_flagList = new axis::domain::collections::FlagCollection();
	_material = NULL;
	_paramList = &aslse::ParameterList::Create();
}

aapc::SectionDefinition& aapc::SectionDefinition::operator=( const SectionDefinition& other )
{
	Copy(other);
	return *this;
}

void aapc::SectionDefinition::Copy( const SectionDefinition& definition )
{
  if (&definition == this) return;
	if (_paramList != NULL) delete _paramList;
	if (_flagList != NULL) delete _flagList;
  if (_material != NULL) _material->Destroy();

	_sectionTypeName = definition.GetSectionTypeName();
	_paramList = &definition._paramList->Clone();
	_flagList = &definition._flagList->Clone();
	if (definition._material == NULL)
	{
		_material = NULL;
	}
	else
	{
		_material = &definition._material->Clone(1);
	}
}

aapc::SectionDefinition::~SectionDefinition( void )
{
	// delete internal variables; material destruction is not our responsibility
 	_paramList->Destroy();
	_flagList->Destroy();
	if (_material != NULL) _material->Destroy();
}

axis::String aapc::SectionDefinition::GetSectionTypeName( void ) const
{
	return (_sectionTypeName);
}

bool aapc::SectionDefinition::operator==( const SectionDefinition& other ) const
{
	return (
		_sectionTypeName.compare(other._sectionTypeName) == 0 &&
		_paramList == other._paramList && 
		_flagList == other._flagList &&
		_material == other._material
		);
}

bool aapc::SectionDefinition::operator!=( const SectionDefinition& other ) const
{
	return !(*this == other);
}

aslse::ParameterList & aapc::SectionDefinition::ParseParameters( 
  const aslse::ParameterValue& params ) const
{
	aslse::ParameterList& pv = aslse::ParameterList::Create();

	if (params.IsArray())
	{	// convert to an array with enumerated items
		aslse::ArrayValue& array = (aslse::ArrayValue&)params;
		for (int i = 0; i < array.Count(); i++)
		{
			aslse::ParameterValue& value = array.Get(i);
			if (value.IsAssignment())	 // convert into key-value pair
			{
				aslse::ParameterAssignment& assignment = (aslse::ParameterAssignment&)value;
				pv.AddParameter(assignment.GetIdName(), assignment.GetRhsValue().Clone());
			}
			else if(value.IsAtomic())
			{	// it is a flag
				aslse::AtomicValue& atom = (aslse::AtomicValue&)value;
				if (atom.IsId() || atom.IsString())	// ok, it is valid
				{
					_flagList->Add(atom.ToString());
				}
				else
				{	// cannot be a number
					throw axis::foundation::ArgumentException();
				}
			}
			else
			{	// whoops, it is something unexpected
				throw axis::foundation::ArgumentException();
			}
		}
	}
	
	return pv;
}

unsigned int aapc::SectionDefinition::PropertyCount( void ) const
{
	return (unsigned int)_paramList->Count();
}

unsigned int aapc::SectionDefinition::FlagCount( void ) const
{
	return (unsigned int)_flagList->Count();
}

aslse::ParameterValue& aapc::SectionDefinition::GetPropertyValue( 
  const axis::String& propertyName ) const
{
	return _paramList->GetParameterValue(propertyName);
}

bool aapc::SectionDefinition::IsPropertyDefined( const axis::String& propertyName ) const
{
	return _paramList->IsDeclared(propertyName);
}

bool aapc::SectionDefinition::IsFlagSet( const axis::String& flag ) const
{
	return _flagList->IsDefined(flag);
}

const axis::domain::materials::MaterialModel& aapc::SectionDefinition::GetMaterial( void ) const
{
	return *_material;
}

void aapc::SectionDefinition::SetMaterial( const adm::MaterialModel& material )
{
	_material = &material;
}

aapc::SectionDefinition& aapc::SectionDefinition::Clone( void ) const
{
	return *new aapc::SectionDefinition(*this);
}

void aapc::SectionDefinition::AddProperty( const axis::String& name, aslse::ParameterValue& value )
{
	if (IsPropertyDefined(name))
	{
		throw axis::foundation::ArgumentException();
	}
	_paramList->AddParameter(name, value.Clone());
}

void aapc::SectionDefinition::AddFlag( const axis::String& flag )
{
	if (IsFlagSet(flag))
	{
		throw axis::foundation::ArgumentException();
	}
	_flagList->Add(flag);
}
