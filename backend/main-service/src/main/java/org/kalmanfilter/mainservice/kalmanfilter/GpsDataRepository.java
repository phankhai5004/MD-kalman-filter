package org.kalmanfilter.mainservice.kalmanfilter;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.repository.JpaRepository;

import java.sql.Timestamp;
import java.util.List;

public interface GpsDataRepository extends JpaRepository<GpsData, Long> {
    Page<GpsData> findByFixtimeBetween(Timestamp fixtime, Timestamp fixtime2, Pageable pageable);

    List<GpsData> findByFixtimeBetween(Timestamp startTimestamp, Timestamp endTimestamp, Sort sort);
}
