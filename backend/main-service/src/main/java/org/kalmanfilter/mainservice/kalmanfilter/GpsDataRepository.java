package org.kalmanfilter.mainservice.kalmanfilter;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.repository.JpaRepository;

import java.sql.Timestamp;
import java.util.List;


public interface GpsDataRepository extends JpaRepository<GpsData, String> {
    Page<GpsData> findByFixtimeBetween(Timestamp timestamp, Timestamp timestamp1, Pageable pageable);
    List<GpsData> findByFixtimeBetween(Timestamp timestamp, Timestamp timestamp1, Sort sort);
}
