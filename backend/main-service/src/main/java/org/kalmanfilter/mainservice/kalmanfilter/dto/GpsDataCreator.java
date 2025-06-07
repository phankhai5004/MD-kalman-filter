package org.kalmanfilter.mainservice.kalmanfilter.dto;

import java.time.Instant;

public record GpsDataCreator(
        Double accuracy,
        Double altitude,
        Double course,
        Instant fixtime,
        Double latitude,
        Double longitude,
        Double speed
) {}
