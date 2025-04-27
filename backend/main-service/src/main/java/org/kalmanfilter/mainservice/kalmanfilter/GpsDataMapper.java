package org.kalmanfilter.mainservice.kalmanfilter;

import org.kalmanfilter.mainservice.kalmanfilter.dto.GpsDataCreator;
import org.kalmanfilter.mainservice.kalmanfilter.dto.GpsDataReponse;
import org.mapstruct.Mapper;
import org.mapstruct.MappingConstants;

@Mapper(componentModel = MappingConstants.ComponentModel.SPRING)
public interface GpsDataMapper {
    GpsDataReponse toGpsDataReponse(GpsData gpsData);
    GpsData toGpsData(GpsDataCreator gpsDataCreator);
}
