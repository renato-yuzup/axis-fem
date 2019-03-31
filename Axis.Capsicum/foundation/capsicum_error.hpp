#pragma once

#define AXIS_ERROR_ID_SCHEDULING_CLASH                      0x306403
#define AXIS_ERROR_MSG_SCHEDULING_CLASH                     _T("Job scheduling definition clashes with enforcements applied to current run. Job execution failed.")

#define AXIS_ERROR_ID_SCHEDULING_MALFORMED_JOB              0x306413
#define AXIS_ERROR_MSG_SCHEDULING_MALFORMED_JOB             _T("Job '%1' cannot be run because the scheduler could not determine a common processing capability for every analysis feature.")

#define AXIS_ERROR_ID_SCHEDULING_UNAVAILABLE_RESOURCE       0x306405
#define AXIS_ERROR_MSG_SCHEDULING_UNAVAILABLE_RESOURCE      _T("Current job cannot be run because no processing resources are available.")

#define AXIS_ERROR_ID_SCHEDULING_FAILED                     0x306416
#define AXIS_ERROR_MSG_SCHEDULING_FAILED                    _T("Job '%1' failed scheduling.")

#define AXIS_ERROR_ID_SCHEDULING_FATAL_ERROR                0x306419
#define AXIS_ERROR_MSG_SCHEDULING_FATAL_ERROR               _T("Fatal error on GPU job scheduling. Analysis process will be aborted.")