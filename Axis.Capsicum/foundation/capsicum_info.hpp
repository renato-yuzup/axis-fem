#pragma once

#define AXIS_INFO_ID_SCHEDULING_ACTIVE                           0x106401
#define AXIS_INFO_MSG_SCHEDULING_ACTIVE                          _T("The execution scheduler is now active.")

#define AXIS_INFO_ID_SCHEDULING_SUBMISSION_OK                    0x106402
#define AXIS_INFO_MSG_SCHEDULING_SUBMISSION_OK                   _T("Job '%1' submission accepted. Preparing to send to execution queue.")

#define AXIS_INFO_ID_SCHEDULING_ENFORCEMENT_APPLIED              0x106414
#define AXIS_INFO_MSG_SCHEDULING_ENFORCEMENT_APPLIED             _T("Scheduling enforcements are in effect. Job scheduling options has been modified.")

#define AXIS_INFO_ID_SCHEDULING_SIMPLE_JOB_FOR_GPU               0x106407
#define AXIS_INFO_MSG_SCHEDULING_SIMPLE_JOB_FOR_GPU              _T("Job is not worth running on GPU. Going to execute on CPU.")

#define AXIS_INFO_ID_SCHEDULING_FINISHED                         0x106417
#define AXIS_INFO_MSG_SCHEDULING_FINISHED                        _T("Execution scheduler finished.")
