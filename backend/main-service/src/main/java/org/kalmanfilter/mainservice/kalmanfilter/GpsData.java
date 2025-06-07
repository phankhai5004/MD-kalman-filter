package org.kalmanfilter.mainservice.kalmanfilter;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Entity
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "gps_data")
public class GpsData {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Double accuracy;
    private Double altitude;
    private Double course;

    private Instant fixtime;
    private Double latitude;
    private Double longitude;
    private Double speed;
}
