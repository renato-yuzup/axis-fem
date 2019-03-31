#ifndef __input_read_error_hpp
#define __input_read_error_hpp

#define AXIS_ERROR_MSG_OPEN_ERROR					_T("Couldn't open input file. Please verify the path is correct and you have sufficient permission to access the file.")
#define AXIS_ERROR_MSG_DEVICE_NOT_READY				_T("Input file opened but it has not yet been signaled for use (device might not be ready perhaps?).")
#define AXIS_ERROR_MSG_OPEN_COMMENT_SYNTAX_ERROR	_T("Comment block at line %1 was not closed prior the end of the file.")
#define AXIS_ERROR_MSG_EOF							_T("Past end of file before comment or group delimiter.")
#define AXIS_ERROR_MSG_UNEXPECTED_END_COMMENT		_T("End comment delimiter without corresponding open delimiter.")
#define AXIS_ERROR_MSG_ENDIF_DELIMITER				_T("Missing #endif.")
#define AXIS_ERROR_MSG_INVALID_ENDIF				_T("Invalid clause -- #endif without #if.")
#define AXIS_ERROR_MSG_INVALID_ENDGROUP				_T("End group delimiter mismatch.")
#define AXIS_ERROR_MSG_OUTOFSTACK					_T("Cannot open include file: input stack full.")
#define AXIS_ERROR_MSG_INVALID_DIRECTIVE			_T("Invalid preprocessor directive.")
#define AXIS_ERROR_MSG_INVALID_INCLUDE_FILE			_T("Cannot open include file -- file not found or might not acessible due to security issues.")
#define AXIS_ERROR_MSG_SYMBOL_REDEFINED				_T("Symbol already defined -- redefinition not allowed.")
#define AXIS_ERROR_MSG_INVALID_SYNTAX				_T("Syntax error.")
#define AXIS_ERROR_MSG_UNEXPECTED_END_BLOCK			_T("Unexpected end block delimiter.")
#define AXIS_ERROR_MSG_UNSUPPORTED_BLOCK			_T("Block not recognized or insufficient parameters.")
#define AXIS_ERROR_MSG_UNEXPECTED_EXPRESSION		_T("Statement seems valid but it is not valid in this context.")
#define AXIS_ERROR_MSG_UNSUPPORTED_OPERATION		_T("This operation is not supported in this context.")

#endif