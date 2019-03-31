#pragma once

#define AXIS_INFO_MSG_RESULT_STRIPPED_EXTENSION								_T("Extension '%1' on file name '%2' was discarded. Resulting file name is '%3'.")


#define AXIS_ERROR_MSG_UNDEFINED_OPEN_PARSER_BEHAVIOR						_T("Undefined parser behavior when reading block header")
#define AXIS_ERROR_ID_UNDEFINED_OPEN_PARSER_BEHAVIOR						0x300507

#define AXIS_ERROR_MSG_UNDEFINED_CLOSE_PARSER_BEHAVIOR						_T("Undefined parser behavior when closing block")
#define AXIS_ERROR_ID_UNDEFINED_CLOSE_PARSER_BEHAVIOR						0x30050D

#define AXIS_ERROR_MSG_UNEXPECTED_PARSER_READ_BEHAVIOR						_T("Parser read fail in current context")
#define AXIS_ERROR_ID_UNEXPECTED_PARSER_READ_BEHAVIOR						0x300508

#define AXIS_ERROR_MSG_GET_PARSER_FAILED									_T("Couldn't open parser due to an unexpected error returned from the provider")
#define AXIS_ERROR_ID_GET_PARSER_FAILED										9004

#define AXIS_ERROR_MSG_END_COMMENT_IN_EXCESS								_T("Too many end comment block delimiter")
#define AXIS_ERROR_ID_END_COMMENT_IN_EXCESS									0x30061D

#define AXIS_ERROR_MSG_UNCLOSED_COMMENT_BLOCK								_T("Unclosed comment block")
#define AXIS_ERROR_ID_UNCLOSED_COMMENT_BLOCK								0x30061E

#define AXIS_ERROR_MSG_MISPLACED_SYMBOL										_T("Misplaced symbol: ")
#define AXIS_ERROR_ID_MISPLACED_SYMBOL										0x300510

#define AXIS_ERROR_MSG_TAIL_IN_EXCESS										_T("Too many block END delimiter")
#define AXIS_ERROR_ID_TAIL_IN_EXCESS										0x30050B

#define AXIS_ERROR_MSG_TAIL_MISSING											_T("Missing END delimiter for block ")
#define AXIS_ERROR_ID_TAIL_MISSING											0x30050E

/* ERROR CODE = 1007 IS RESERVED */

#define AXIS_ERROR_MSG_OUTPUT_FILE_CONFLICT							_T("There is a conflict between two or more output chains trying to use the same file: %1")
#define AXIS_ERROR_ID_OUTPUT_FILE_CONFLICT							0x300565

#define AXIS_ERROR_MSG_OUTPUT_APPEND_ERROR							_T("It is not possible to concatenate results from different steps for the output format '%1' with the specified set of parameters.")
#define AXIS_ERROR_ID_OUTPUT_APPEND_ERROR							0x300569

#define AXIS_ERROR_MSG_OUTPUT_APPEND_VIOLATION							_T("Although in append operation, two or more chains cannot share the same output file in the same step.")
#define AXIS_ERROR_ID_OUTPUT_APPEND_VIOLATION							0x30056A

#define AXIS_ERROR_MSG_STATEMENT_TOO_LONG									_T("Statement too long to be processed")
#define AXIS_ERROR_ID_STATEMENT_TOO_LONG									0x30050F

#define AXIS_ERROR_MSG_INVALID_CHAR_SEQUENCE								_T("Invalid character or character sequence: ")
#define AXIS_ERROR_ID_INVALID_CHAR_SEQUENCE									0x300511

#define AXIS_ERROR_MSG_UNPROCESSED_BLOCK									_T("Block not processed due to one or more errors")
#define AXIS_ERROR_ID_UNPROCESSED_BLOCK										1010

#define AXIS_ERROR_MSG_BLOCK_DELIMITER_MISMATCH								_T("Block delimiter does not match with corresponding header. Expected: ")
#define AXIS_ERROR_ID_BLOCK_DELIMITER_MISMATCH								0x30050C

#define AXIS_ERROR_MSG_ENDIF_MISSING										_T("Missing @endif delimiter")
#define AXIS_ERROR_ID_ENDIF_MISSING											0x300601

#define AXIS_ERROR_MSG_ENDIF_IN_EXCESS										_T("Too many @endif directives -- no matching @if were found")
#define AXIS_ERROR_ID_ENDIF_IN_EXCESS										0x300607

#define AXIS_ERROR_MSG_ELSE_WITHOUT_IF										_T("@else without @if (no matching @if were found)")
#define AXIS_ERROR_ID_ELSE_WITHOUT_IF										0x300608

#define AXIS_ERROR_MSG_UNKNOWN_DIRECTIVE									_T("Unknown preprocessor directive")
#define AXIS_ERROR_ID_UNKNOWN_DIRECTIVE										0x300602

#define AXIS_ERROR_MSG_DIRECTIVE_SYNTAX_ERROR								_T("Invalid directive syntax")
#define AXIS_ERROR_ID_DIRECTIVE_SYNTAX_ERROR								0x300605

#define AXIS_ERROR_MSG_REDEFINED_DIRECTIVE_FLAG								_T("Redefined directive flag: ")
#define AXIS_ERROR_ID_REDEFINED_DIRECTIVE_FLAG								2001

#define AXIS_ERROR_MSG_UNKNOWN_BLOCK										_T("Unknown block in current context: ")
#define AXIS_ERROR_ID_UNKNOWN_BLOCK											0x300509

#define AXIS_ERROR_MSG_REDEFINED_SYMBOL										_T("Redefined symbol: ")
#define AXIS_ERROR_ID_REDEFINED_SYMBOL										2001

#define AXIS_ERROR_MSG_MISSING_BLOCK_PARAM									_T("Missing block parameter: ")
#define AXIS_ERROR_ID_MISSING_BLOCK_PARAM									2002

#define AXIS_ERROR_MSG_UNKNOWN_BLOCK_PARAM									_T("Unknown block parameter: ")
#define AXIS_ERROR_ID_UNKNOWN_BLOCK_PARAM									0x300501

#define AXIS_ERROR_MSG_BLOCK_PARAM_MUTUALLY_EXCLUSIVE						_T("One or more specified parameters cannot be used at the same time: ")
#define AXIS_ERROR_ID_BLOCK_PARAM_MUTUALLY_EXCLUSIVE						2004

#define AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE								_T("Invalid value type for block parameter: ")
#define AXIS_ERROR_ID_INVALID_PARAM_VALUE_TYPE								2005

#define AXIS_ERROR_MSG_INVALID_VALUE_TYPE									_T("Type mismatch: ")
#define AXIS_ERROR_ID_INVALID_VALUE_TYPE									2006

#define AXIS_ERROR_MSG_NODE_PARSER_INVALID_ID								_T("Invalid node identifier")
#define AXIS_ERROR_ID_NODE_PARSER_INVALID_ID								2007

#define AXIS_ERROR_MSG_NODE_PARSER_DUPLICATED_ID							_T("Duplicated node identifier: ")
#define AXIS_ERROR_ID_NODE_PARSER_DUPLICATED_ID								2008

#define AXIS_ERROR_MSG_NODESET_PARSER_DUPLICATED_ID							_T("Duplicated node set identifier: ")
#define AXIS_ERROR_ID_NODESET_PARSER_DUPLICATED_ID							2009

#define AXIS_ERROR_MSG_VALUE_OUT_OF_RANGE									_T("Value out of range: ")
#define AXIS_ERROR_ID_NODESET_PARSER_OUT_OF_RANGE							2010

#define AXIS_ERROR_MSG_NODESET_PARSER_INVALID_RANGE							_T("Invalid range: ")
#define AXIS_ERROR_ID_NODESET_PARSER_INVALID_RANGE							2011

#define AXIS_ERROR_MSG_PART_PARSER_INVALID_SECTION							_T("Unrecognized element type, missing parameters or invalid parameters supplied.")
#define AXIS_ERROR_ID_PART_PARSER_INVALID_SECTION							2013

#define AXIS_ERROR_MSG_PART_PARSER_INVALID_MATERIAL							_T("Unrecognized material type, missing parameters or invalid parameters supplied.")
#define AXIS_ERROR_ID_PART_PARSER_INVALID_MATERIAL							2014

#define AXIS_ERROR_MSG_PART_PARSER_SECTION_REDEFINITION						_T("Part definition for element set %1 has already been defined. Cannot redefine it.")
#define AXIS_ERROR_ID_PART_PARSER_SECTION_REDEFINITION						2015

#define AXIS_ERROR_MSG_INVALID_DECLARATION									_T("Unrecognized statement syntax in this context.")
#define AXIS_ERROR_ID_INVALID_DECLARATION									2016

#define AXIS_ERROR_MSG_ELEMENT_PARSER_INVALID_DECLARATION					_T("Invalid element declaration syntax or insufficient nodes to build specified element type.")
#define AXIS_ERROR_ID_ELEMENT_PARSER_INVALID_DECLARATION					2016

#define AXIS_ERROR_MSG_ELEMENT_PARSER_DUPLICATED_ID							_T("Duplicated element identifier: ")
#define AXIS_ERROR_ID_ELEMENT_PARSER_DUPLICATED_ID							2017

#define AXIS_ERROR_MSG_ELEMENT_PARSER_INCOMPATIBLE_NODE						_T("Node %1 is not compatible with the definition of element %2. Degrees of freedom required differ.")
#define AXIS_ERROR_ID_ELEMENT_PARSER_INCOMPATIBLE_NODE						0x300534

#define AXIS_ERROR_MSG_UNKNOWN_TIME_CONTROL_ALGORITHM						_T("Unrecognized time control algorithm or incorrect parameters were supplied. Analysis configuration could not be inferred.")
#define AXIS_ERROR_ID_UNKNOWN_TIME_CONTROL_ALGORITHM						0x30054F

#define AXIS_ERROR_MSG_UNKNOWN_SOLVER_TYPE									_T("Unrecognized solver type or incorrect parameters were supplied. Analysis configuration could not be inferred.")
#define AXIS_ERROR_ID_UNKNOWN_SOLVER_TYPE									0x300550

#define AXIS_ERROR_MSG_INVALID_SNAPSHOT_TIME								_T("Invalid snapshot time or range.")
#define AXIS_ERROR_ID_INVALID_SNAPSHOT_TIME									0x300551

#define AXIS_ERROR_MSG_OVERLAPPED_SNAPSHOT									_T("Supplied snapshot time overlaps with others previously declared for the step.")
#define AXIS_ERROR_ID_OVERLAPPED_SNAPSHOT									0x300552

#define AXIS_ERROR_MSG_SNAPSHOT_OUT_OF_RANGE								_T("Snapshot time out of the step time frame.")
#define AXIS_ERROR_ID_SNAPSHOT_OUT_OF_RANGE									0x300553

#define AXIS_ERROR_MSG_SNAPSHOT_LOCKED										_T("Declaring a fixed amount of snapshots must be a unique statement in the block.")
#define AXIS_ERROR_ID_SNAPSHOT_LOCKED										0x300554

#define AXIS_ERROR_MSG_NODE_NOT_FOUND										_T("Node not found: ")
#define AXIS_ERROR_ID_NODE_NOT_FOUND										2018

#define AXIS_ERROR_MSG_ELEMENT_NOT_FOUND									_T("Element not found: ")
#define AXIS_ERROR_ID_ELEMENT_NOT_FOUND										2019

#define AXIS_ERROR_MSG_NODESET_NOT_FOUND									_T("Node set not found: ")
#define AXIS_ERROR_ID_NODESET_NOT_FOUND										0x300522

#define AXIS_ERROR_MSG_ELEMENTSET_NOT_FOUND									_T("Element set not found: ")
#define AXIS_ERROR_ID_ELEMENTSET_NOT_FOUND									0x300517

#define AXIS_ERROR_MSG_CURVE_NOT_FOUND										_T("Curve not found: ")
#define AXIS_ERROR_ID_CURVE_NOT_FOUND										0x300537

#define AXIS_ERROR_MSG_ELEMENTSET_PARSER_DUPLICATED_ID						_T("Duplicated element set identifier: ")
#define AXIS_ERROR_ID_ELEMENTSET_PARSER_DUPLICATED_ID						2023

#define AXIS_ERROR_MSG_PART_NOT_FOUND										_T("Section definition not found for the following element set: ")
#define AXIS_ERROR_ID_PART_NOT_FOUND										2040

#define AXIS_ERROR_MSG_BOUNDARY_CONDITION_REDEFINED							_T("Boundary condition reapplied to node set %1 or node set overlaps with nodes having boundary conditions already set.")
#define AXIS_ERROR_ID_BOUNDARY_CONDITION_REDEFINED							2041

#define AXIS_ERROR_MSG_DUPLICATED_BLOCK										_T("Block must be unique; two or more instances were detected.")
#define AXIS_ERROR_ID_DUPLICATED_BLOCK										2042

#define AXIS_ERROR_MSG_COLLECTOR_INVALID_SET								_T("Specified grouping data method for this collector does not allow multiple items in the collection set ")
#define AXIS_ERROR_ID_COLLECTOR_INVALID_SET									0x300546

#define AXIS_ERROR_MSG_RESULT_DUPLICATED_STREAM								_T("Cannot create multiple result chains writing to the same output file: ")
#define AXIS_ERROR_ID_RESULT_DUPLICATED_STREAM								2042

#define AXIS_ERROR_MSG_RESULT_ISOLATION_VIOLATION							_T("At least one result collector in the chain requires exclusive access to output file (that is, should be alone in chain declaration). Output declaration failed.")
#define AXIS_ERROR_ID_RESULT_ISOLATION_VIOLATION							2042

#define AXIS_ERROR_MSG_RESULT_FORMAT_MISMATCH								_T("Although all collectors in the chain output to the same format set specified in the block header, one more collectors disagree in which specific format to use (ASCII report format or binary database, for example. Collectors chain processing was aborted.")
#define AXIS_ERROR_ID_RESULT_FORMAT_MISMATCH								2042

#define AXIS_ERROR_MSG_UNRECOGNIZED_OUTPUT_FORMAT							_T("Unrecognized output format '%1' with the supplied format arguments.")
#define AXIS_ERROR_ID_UNRECOGNIZED_OUTPUT_FORMAT							0x30055F

#define AXIS_ERROR_MSG_RESULT_CREATE_STREAM_FAILED							_T("Couldn't get access to file '%1'. Check if it is a valid file name and all folders in the specified location exist.")
#define AXIS_ERROR_ID_RESULT_CREATE_STREAM_FAILED							2042

#define AXIS_ERROR_MSG_INCLUDE_FILE_NOT_FOUND								_T("Input file not found or cannot be accessed: ")
#define AXIS_ERROR_ID_INCLUDE_FILE_NOT_FOUND								0x300604

#define AXIS_ERROR_MSG_INCLUDE_FILE_IO_ERROR								_T("I/O error occurred while processing file: ")
#define AXIS_ERROR_ID_INCLUDE_FILE_IO_ERROR									0x30061F

#define AXIS_ERROR_MSG_INPUT_STACK_OVERFLOW									_T("Too many nested include files; input stack overflow, operation aborted.")
#define AXIS_ERROR_ID_INPUT_STACK_OVERFLOW									0x300603

#define AXIS_ERROR_MSG_INFINITE_READ										_T("Unexpected behavior of input parsers resulted in successive retries to successfully parse the input file. After %1 tries we gave up.")
#define AXIS_ERROR_ID_INFINITE_READ											8004

#define AXIS_ERROR_MSG_TOO_MANY_ERRORS										_T("Too many errors occurred. Operation aborted.")
#define AXIS_ERROR_ID_TOO_MANY_ERRORS										8005
