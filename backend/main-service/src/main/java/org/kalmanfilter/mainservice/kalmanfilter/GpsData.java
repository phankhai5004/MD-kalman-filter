package org.kalmanfilter.mainservice.kalmanfilter;

import jakarta.persistence.*;
import lombok.*;

import java.sql.Timestamp;

@Entity
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "gps_data")
public class GpsData {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String id;

    private Double accuracy;
    private Double altitude;
    private Double course;

    @Column(name = "fixtime")
    private Timestamp fixtime;
    private Double latitude;
    private Double longitude;
    private Double speed;
}
