#pragma once

#define AXIS_WARN_ID_SCHEDULING_NO_GPU_FOUND              0x206404
#define AXIS_WARN_MSG_SCHEDULING_NO_GPU_FOUND             _T("Job scheduling option was set to GPU, but no compatible GPUs were found in the system. Falling back to CPU.")

#define AXIS_WARN_ID_SCHEDULING_GPU_BUSY                  0x206404
#define AXIS_WARN_MSG_SCHEDULING_GPU_BUSY                 _T("Job scheduling option was set to GPU, but GPU resources appear to be very busy. Falling back to CPU.")

#define AXIS_WARN_ID_SCHEDULING_DEVICE_NOT_RELEASED       0x206418
#define AXIS_WARN_MSG_SCHEDULING_DEVICE_NOT_RELEASED      _T("One or more GPU devices could not be marked as free. Current execution will continue, but future jobs might not have access to these devices.")
