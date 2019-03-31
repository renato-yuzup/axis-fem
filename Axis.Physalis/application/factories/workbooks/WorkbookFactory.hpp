#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "AxisString.hpp"
#include "application/fwd/output_collectors.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"

namespace axis { namespace application { namespace factories { namespace workbooks {

/**
 * Represents an object capable of building a specific workbook format.
 */
class AXISPHYSALIS_API WorkbookFactory
{
public:
	WorkbookFactory(void);
	virtual ~WorkbookFactory(void);

  /**
   * Destroys this object.
   */
	virtual void Destroy(void) const = 0;

  /**
   * Determines if this object can build the specified workbook format.
   *
   * @param formatName      Name of the workbook format.
   * @param formatArguments The required and optional format arguments.
   *
   * @return true if it can build, false otherwise.
   */
	virtual bool CanBuild(const axis::String& formatName, 
    const axis::services::language::syntax::evaluation::ParameterList& formatArguments) const = 0;

  /**
   * Builds a new workbook format.
   *
   * @param formatName      Name of the workbook format.
   * @param formatArguments The required and optional format arguments.
   *
   * @return A workbook object.
   */
	virtual axis::application::output::workbooks::ResultWorkbook& BuildWorkbook(
    const axis::String& formatName, 
    const axis::services::language::syntax::evaluation::ParameterList& formatArguments) = 0;
};

} } } } // namespace axis::application::factories::workbooks
