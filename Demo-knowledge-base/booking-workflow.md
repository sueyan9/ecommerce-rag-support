# Booking Workflow

Bookings move through four simple states in this demo system: draft, pending confirmation, confirmed, and completed.

## Draft booking

A draft booking can be created before all details are known. The staff member should capture branch, practitioner, appointment type, and preferred time range.

## Pending confirmation

If the clinic is waiting for confirmation from the patient or practitioner, the booking should remain in pending confirmation. Pending bookings still appear in schedule views but are visually marked to prevent double allocation.

## Confirmed booking

Confirmed bookings require a scheduled time, assigned practitioner, and contactable patient profile. Reminder messages should only be sent after the booking is confirmed.

## Cancellation and rescheduling

Cancelled bookings are retained for reporting. Rescheduling creates a new time assignment while preserving the original audit trail.
