#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "services/management/Provider.hpp"
#include "application/factories/Collectors/CollectorFactory.hpp"
#include "application/fwd/parsing.hpp"

namespace axis { namespace application { namespace locators {

/**
 * Provides means to determine if there is a capable factory to build a specific collector.
 *
 * @sa axis::services::management::Provider
 */
class AXISPHYSALIS_API CollectorFactoryLocator : public axis::services::management::Provider
{
public:
  CollectorFactoryLocator(void);
	~CollectorFactoryLocator(void);

  /**
   * Registers a new collector factory.
   *
   * @param [in,out] factory The factory.
  **/
	void RegisterFactory(axis::application::factories::collectors::CollectorFactory& factory);

  /**
   * Unregisters a collector factory.
   *
   * @param [in,out] factory The factory.
  **/
	void UnregisterFactory(axis::application::factories::collectors::CollectorFactory& factory);

  /**
   * Checks if there is any factory that can parse the specified statement.
   *
   * @param formatName Name of the corresponding workbook format.
   * @param begin      Iterator pointing to the beginning of the statement.
   * @param end        Iterator pointing to the end of the statement.
   *
   * @return The result of parsing operation.
  **/
	axis::services::language::parsing::ParseResult TryParse(
                        const axis::String& formatName, 
                        const axis::services::language::iterators::InputIterator& begin, 
                        const axis::services::language::iterators::InputIterator& end);

  /**
   * Parse and build.
   *
   * @param model            The numerical model where collection will occur.
   * @param [in,out] context The current parse context.
   * @param formatName       Name of the corresponding workbook format.
   * @param begin            Iterator pointing to the beginning of the statement.
   * @param end              Iterator pointing to the end of the statement.
   *
   * @return .
  **/
	axis::application::factories::collectors::CollectorBuildResult ParseAndBuild(
    const axis::domain::analyses::NumericalModel& model, 
                        axis::application::parsing::core::ParseContext& context, 
                        const axis::String& formatName, 
                        const axis::services::language::iterators::InputIterator& begin, 
                        const axis::services::language::iterators::InputIterator& end);

  virtual const char * GetFeaturePath( void ) const;
  virtual const char * GetFeatureName( void ) const;
protected:
  virtual void DoOnPostProcessRegistration(
    axis::services::management::GlobalProviderCatalog& rootManager);
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } }
