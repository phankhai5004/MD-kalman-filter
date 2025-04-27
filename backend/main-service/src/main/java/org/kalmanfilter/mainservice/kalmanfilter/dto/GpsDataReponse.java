package org.kalmanfilter.mainservice.kalmanfilter.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.sql.Timestamp;

@Getter
@Setter
@Builder
public class GpsDataReponse {
    private Double accuracy;
    private Double altitude;
    private Double course;

    private Timestamp fixtime;
    private Double latitude;
    private Double longitude;
    private Double speed;
}
