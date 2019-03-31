#pragma once
#include "application/factories/collectors/CollectorFactory.hpp"
#include "services/management/Provider.hpp"

namespace axis { namespace application { 

namespace output { namespace collectors {
class NodeSetCollector;
} }

namespace factories { namespace collectors {

class TextReportNodeCollectorFactory : public CollectorFactory
{
public:
  enum XDirectionState // states for capturing field values in X-direction
  {
    kXDirectionEnabled,
    kXDirectionDisabled
  };
  enum YDirectionState // states for capturing field values in Y-direction
  {
    kYDirectionEnabled,
    kYDirectionDisabled
  };
  enum ZDirectionState // states for capturing field values in Z-direction
  {
    kZDirectionEnabled,
    kZDirectionDisabled
  };
  enum XXDirectionState // states for capturing tensor values in XX-direction
  {
    kXXDirectionEnabled,
    kXXDirectionDisabled
  };
  enum YYDirectionState // states for capturing tensor values in YY-direction
  {
    kYYDirectionEnabled,
    kYYDirectionDisabled
  };
  enum ZZDirectionState // states for capturing tensor values in ZZ-direction
  {
    kZZDirectionEnabled,
    kZZDirectionDisabled
  };
  enum XYDirectionState // states for capturing tensor values in XY-direction
  {
    kXYDirectionEnabled,
    kXYDirectionDisabled
  };
  enum YZDirectionState // states for capturing tensor values in YZ-direction
  {
    kYZDirectionEnabled,
    kYZDirectionDisabled
  };
  enum XZDirectionState // states for capturing tensor values in XZ-direction
  {
    kXZDirectionEnabled,
    kXZDirectionDisabled
  };

  TextReportNodeCollectorFactory(void);
  virtual ~TextReportNodeCollectorFactory(void);

  virtual void Destroy( void ) const;

  /**
   * Tries to parse and resolve a statement for any known collectors of this factory.
   *
   * @param formatName Argument ignored; just for interface compatibility.
   * @param begin      The iterator pointing to the beginning of the statement.
   * @param end        The iterator pointing to the end of the statement.
   *
   * @return The best match result found for this statement.
   */
  virtual axis::services::language::parsing::ParseResult TryParse( 
                                     const axis::String& formatName, 
                                     const axis::services::language::iterators::InputIterator& begin, 
                                     const axis::services::language::iterators::InputIterator& end );

  /**
   * Parses a statement and build the corresponding collector, which is one of those
   * known by this factory.
   *
   * @param formatName       Argument ignored; just for interface compatibility.
   * @param begin            The iterator pointing to the beginning of the statement.
   * @param end              The iterator pointing to the end of the statement.
   * @param model            The analysis numerical model.
   * @param [in,out] context The parsing context.
   *
   * @return An object holding information about the parsing result and possibly the created collector.
   */
  virtual CollectorBuildResult ParseAndBuild( const axis::String& formatName, 
                                     const axis::services::language::iterators::InputIterator& begin, 
                                     const axis::services::language::iterators::InputIterator& end, 
                                     const axis::domain::analyses::NumericalModel& model, 
                                     axis::application::parsing::core::ParseContext& context );
private:
  class Pimpl;

  TextReportNodeCollectorFactory(const TextReportNodeCollectorFactory&);
  TextReportNodeCollectorFactory& operator = (const TextReportNodeCollectorFactory&);

  Pimpl *pimpl_;
};

} } } } // namespace axis::application::factories::collectors
