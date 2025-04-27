package org.kalmanfilter.mainservice.kalmanfilter;

import lombok.RequiredArgsConstructor;
import org.kalmanfilter.mainservice.dtos.PageResponse;
import org.kalmanfilter.mainservice.dtos.Response;
import org.kalmanfilter.mainservice.kalmanfilter.dto.GpsDataCreator;
import org.kalmanfilter.mainservice.kalmanfilter.dto.GpsDataReponse;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.sql.Timestamp;
import java.time.LocalDateTime;

@RestController
@RequestMapping("/gps")
@RequiredArgsConstructor
public class GpsDataController {
    private final GpsDataService gpsDataService;

    @PostMapping
    public Response<Void> createOne(
            @RequestBody GpsDataCreator gpsDataCreator
    ) {
        gpsDataService.createOne(gpsDataCreator);

        return Response.<Void>builder()
                .success(true)
                .build();
    }

    @GetMapping
    public Response<PageResponse<GpsDataReponse>> readAll(
            @RequestParam(defaultValue = "1", required = false)
            int page,

            @RequestParam(defaultValue = "10", required = false)
            int size,

            @RequestParam(defaultValue = "true")
            boolean asc,

            @RequestParam(defaultValue = "true")
            boolean latest
    ) {
        // Handle sorting
        Sort.Direction direction = asc ? Sort.Direction.ASC : Sort.Direction.DESC;
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, "fixtime"));

        // Handle latest filter
        if (latest) {
            LocalDateTime to = LocalDateTime.now();
            LocalDateTime from = to.minusDays(7);
            return Response.<PageResponse<GpsDataReponse>>builder()
                    .success(true)
                    .data(gpsDataService.getLatest(from, to, pageable))
                    .build();
        }

        // Default: return all
        return Response.<PageResponse<GpsDataReponse>>builder()
                .success(true)
                .data(gpsDataService.getAll(pageable))
                .build();
    }

    @GetMapping("/range")
    public Response<PageResponse<GpsDataReponse>> readByFixTimeBetween(
            @RequestParam("start")
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
            LocalDateTime start,

            @RequestParam("end")
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
            LocalDateTime end,

            @RequestParam(defaultValue = "true")
            boolean asc,

            @RequestParam(defaultValue = "true")
            boolean latest
    ) {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime defaultStart = now.minusDays(7);

        LocalDateTime startTime = (latest || start == null) ? defaultStart : start;
        LocalDateTime endTime = (latest || end == null) ? now : end;

        Timestamp startTimestamp = Timestamp.valueOf(startTime);
        Timestamp endTimestamp = Timestamp.valueOf(endTime);

        Sort sort = Sort.by(asc ? Sort.Direction.ASC : Sort.Direction.DESC, "fixtime");

        return Response.<PageResponse<GpsDataReponse>>builder()
                .success(true)
                .data(gpsDataService.getAllByFixTimeBetween(startTimestamp, endTimestamp, sort))
                .build();
    }
}
