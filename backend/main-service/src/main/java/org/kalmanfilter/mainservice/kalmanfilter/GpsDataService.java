package org.kalmanfilter.mainservice.kalmanfilter;

import lombok.RequiredArgsConstructor;
import org.kalmanfilter.mainservice.dtos.PageResponse;
import org.kalmanfilter.mainservice.kalmanfilter.dto.GpsDataCreator;
import org.kalmanfilter.mainservice.kalmanfilter.dto.GpsDataReponse;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class GpsDataService {
    private final GpsDataMapper gpsDataMapper;
    private final GpsDataRepository gpsDataRepository;

    private final KafkaTemplate<String, Object> kafkaTemplate;

    public void createOne(
            GpsDataCreator gpsDataCreator
    ) {
        GpsData gpsData = gpsDataMapper.toGpsData(gpsDataCreator);

        gpsData = gpsDataRepository.save(gpsData);
        kafkaTemplate.send("gps-data", gpsData);
    }

    public PageResponse<GpsDataReponse> getAll(
            Pageable pageable
    ) {
        Page<GpsData> gpsDataPage = gpsDataRepository.findAll(pageable);
        List<GpsDataReponse> gpsDataList = gpsDataPage.getContent()
                .stream().map(gpsDataMapper::toGpsDataReponse).toList();

        return PageResponse.<GpsDataReponse>builder()
                .items(gpsDataList)
                .records(gpsDataPage.getNumberOfElements())
                .page(pageable.getPageNumber())
                .totalPages(gpsDataPage.getTotalPages())
                .build();
    }

    public PageResponse<GpsDataReponse> getLatest(
            LocalDateTime from,
            LocalDateTime to,
            Pageable pageable
    ) {
        Page<GpsData> gpsDataPage = gpsDataRepository.findByFixtimeBetween(
                Timestamp.valueOf(from),
                Timestamp.valueOf(to),
                pageable
        );

        List<GpsDataReponse> gpsDataList = gpsDataPage.getContent()
                .stream()
                .map(gpsDataMapper::toGpsDataReponse)
                .toList();

        return PageResponse.<GpsDataReponse>builder()
                .items(gpsDataList)
                .records(gpsDataPage.getNumberOfElements())
                .page(pageable.getPageNumber())
                .totalPages(gpsDataPage.getTotalPages())
                .build();
    }

    public PageResponse<GpsDataReponse> getAllByFixTimeBetween(
            Timestamp startTimestamp,
            Timestamp endTimestamp,
            Sort sort
    ) {
        List<GpsDataReponse> gpsDataList = gpsDataRepository.findByFixtimeBetween(startTimestamp,endTimestamp, sort)
                .stream().map(gpsDataMapper::toGpsDataReponse).toList();

        return PageResponse.<GpsDataReponse>builder()
                .items(gpsDataList)
                .records(gpsDataList.size())
                .build();
    }
}
