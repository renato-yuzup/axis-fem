#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/fwd/output_collectors.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/management/Provider.hpp"

namespace axis { namespace application {

namespace factories { namespace workbooks {
class WorkbookFactory;
} } // namespace ~::factories::workbooks

namespace locators {

/**
 * Provides means to determine if there is a factory capable of 
 * building a specific workbook format.
 *
 * @sa axis::services::management::Provider
**/
class AXISPHYSALIS_API WorkbookFactoryLocator : public axis::services::management::Provider
{
public:
	WorkbookFactoryLocator(void);
	~WorkbookFactoryLocator(void);

  /**
   * Registers a new workbook factory.
   *
   * @param [in,out] factory The factory.
  **/
	void RegisterFactory(axis::application::factories::workbooks::WorkbookFactory& factory);

  /**
   * Unregisters a factory.
   *
   * @param [in,out] factory The factory.
  **/
	void UnregisterFactory(axis::application::factories::workbooks::WorkbookFactory& factory);

  /**
   * Determines there is a factory capable of building a specific format.
   *
   * @param formatName      Name of the workbook format.
   * @param formatArguments The format arguments.
   *
   * @return true if a factory claims to be able to build, false otherwise.
  **/
	bool CanBuild(const axis::String& formatName, 
    const axis::services::language::syntax::evaluation::ParameterList& formatArguments) const;

  /**
   * Builds a new workbook format object.
   *
   * @param formatName      Name of the workbook format.
   * @param formatArguments The format arguments.
   *
   * @return The workbook.
  **/
	axis::application::output::workbooks::ResultWorkbook& BuildWorkbook(
    const axis::String& formatName, 
    const axis::services::language::syntax::evaluation::ParameterList& formatArguments);

  virtual const char * GetFeaturePath( void ) const;
  virtual const char * GetFeatureName( void ) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::application::locators
