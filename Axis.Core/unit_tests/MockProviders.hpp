#pragma once

#include "AxisString.hpp"
#include "application/locators/ElementParserLocator.hpp"
#include "application/locators/MaterialFactoryLocator.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "domain/materials/MaterialModel.hpp"

using namespace axis::foundation;
using namespace axis::application::factories::parsers;
using namespace axis::application::locators;
using namespace axis::application::parsing::parsers;
using namespace axis::application::parsing::core;
using namespace axis::domain::materials;

class TestMaterial : public MaterialModel
{
public:
  TestMaterial(void);
	bool IsTestMaterial(void) const;
	virtual void Destroy( void ) const;
	virtual axis::foundation::blas::DenseMatrix& GetMaterialTensor( void ) const;
	virtual MaterialModel& Clone( int numPoints ) const;
	virtual void UpdateStresses( 
    axis::domain::physics::UpdatedPhysicalState& updatedState,
    const axis::domain::physics::InfinitesimalState& elementInfo, 
    const axis::domain::analyses::AnalysisTimeline& timeInfo,
    int materialPointIndex );
	virtual real GetWavePropagationSpeed( void ) const;
  virtual real GetBulkModulus( void ) const;
  virtual real GetShearModulus( void ) const;
  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;
};

class MockContext : public ParseContext
{
private:
	int _eventsCount;
	int _crossRefCount;
public:
	MockContext(void);
	virtual void AddUndefinedCustomSymbol( const axis::String& name, 
    const axis::String& type );
	virtual void DefineCustomSymbol( const axis::String& name, 
    const axis::String& type );
	virtual RunMode GetRunMode( void ) const;
	virtual void RegisterEvent( axis::services::messaging::EventMessage& event );
	int GetRegisteredEventsCount(void) const;
	void ClearRegisteredEvents(void);
	int GetCrossRefCount(void) const;
	void ClearCrossRefs(void);
};

class MockProvider : public BlockProvider
{
public:
	virtual const char *GetFeaturePath(void) const;
	virtual const char *GetFeatureName(void) const;
	virtual void PostProcessRegistration(
    axis::services::management::GlobalProviderCatalog& manager);
	virtual void UnloadModule(
    axis::services::management::GlobalProviderCatalog& manager);
	virtual bool CanParse( const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
};

class MockElementProvider : public ElementParserLocator
{
private:
	mutable int _failedCount;
	mutable int _sucessfulCount;
public:
	MockElementProvider(void);
	virtual const char *GetFeaturePath(void) const;
	virtual const char *GetFeatureName(void) const;
	virtual void PostProcessRegistration(
    axis::services::management::GlobalProviderCatalog& manager);
	virtual void UnloadModule(
    axis::services::management::GlobalProviderCatalog& manager);
	virtual void RegisterFactory( ElementParserFactory& factory );
	virtual void UnregisterFactory( ElementParserFactory& factory );
	virtual bool CanParse(const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
  virtual bool CanBuildElement( 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition
    ) const;
	virtual axis::application::parsing::parsers::BlockParser& BuildParser(
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual ElementParserFactory& GetFactory( 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition);
	virtual ElementParserFactory& GetFactory( 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition
    ) const;
	int GetFailedElementBuildQueryCount(void) const;
	int GetSuccessfulElementBuildQueryCount(void) const;
	void ResetCounters(void);
	virtual axis::application::parsing::parsers::BlockParser& 
    BuildVoidParser( void ) const;
};

class MockMaterialProvider : public MaterialFactoryLocator
{
private:
	mutable int _incorrectMaterialQueryCount;
	mutable int _invalidParamsQueryCount;
	mutable int _successfulQueryCount;

public:
	MockMaterialProvider(void);
	virtual const char *GetFeaturePath(void) const;
	virtual const char *GetFeatureName(void) const;
	virtual void PostProcessRegistration(
    axis::services::management::GlobalProviderCatalog& manager);
	virtual void UnloadModule(
    axis::services::management::GlobalProviderCatalog& manager);
	virtual void RegisterFactory( 
    axis::application::factories::materials::MaterialFactory& factory );
	virtual void UnregisterFactory( 
    axis::application::factories::materials::MaterialFactory& factory );
	virtual bool CanBuild( const axis::String& modelName, 
    const axis::services::language::syntax::evaluation::ParameterList& params 
    ) const;
	virtual axis::domain::materials::MaterialModel& BuildMaterial( 
    const axis::String& modelName, 
    const axis::services::language::syntax::evaluation::ParameterList& params );
	int GetIncorrectMaterialQueryCount(void) const;
	int GetInvalidParamsQueryCount(void) const;
	int GetSuccessfulQueryCount(void) const;
	void ResetCounters(void);
};
