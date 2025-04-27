package org.kalmanfilter.mainservice.kalmanfilter.dto;

import java.sql.Timestamp;

public record GpsDataCreator(
        Double accuracy,
        Double altitude,
        Double course,
        Timestamp fixtime,
        Double latitude,
        Double longitude,
        Double speed
) {}
