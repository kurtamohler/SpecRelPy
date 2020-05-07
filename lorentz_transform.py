import numpy

# position          1-D or greater
# time              0-D or greater, shape must equal position.shape[:-1]
# velocity          1-D, shape must equal position.shape[-1]
# speed_of_light    0-D
def lorentz_transform(position, time, velocity, speed_of_light=300_000_000):
    position_shape = position.shape
    num_spatial_dims = position_shape[-1]
    time_shape = time.shape

    if len(position_shape) != len(time_shape) + 1:
        raise ValueError("position must have 1 more dim than time")

    if position_shape[:-1] != time_shape:
        raise ValueError("position.shape[:-1] must equal time.shape")

    num_events = numpy.prod(time_shape)

    position_squeezed = position.reshape(num_events, num_spatial_dims)
    time_squeezed = time.reshape(num_events)

    position_transformed, time_transformed = lorentz_transform_inner(position_squeezed, time_squeezed, velocity, speed_of_light)

    return position_transformed.reshape(position_shape), time_transformed.reshape(time_shape)


def lorentz_transform_inner(position, time, velocity, speed_of_light=300_000_000):
    speed_of_light_squ = numpy.double(speed_of_light)**2

    position = numpy.array(position, dtype=numpy.double)
    time = numpy.array(time, dtype=numpy.double)
    velocity = numpy.array(velocity, dtype=numpy.double)

    speed = numpy.linalg.norm(velocity, axis=-1)
    if speed == 0:
        return position, time

    if speed >= speed_of_light:
        raise ValueError("magnitude of velocity, %f, must be less than the speed of light, %f" % (
            speed, speed_of_light
        ))

    velocity_direction = velocity / speed

    lorentz_factor = numpy.double(1.0) / (numpy.double(1.0) - (speed * speed) / speed_of_light_squ)**numpy.double(0.5)

    # Project the position along the velocity
    projected_position = numpy.sum(position * velocity_direction, axis=-1)

    projected_position_transformed = lorentz_factor * (projected_position - speed * time)
    time_transformed = lorentz_factor * (time - speed * projected_position / speed_of_light_squ)

    # TODO: I don't like having two different operations here, there must be a generalized way to do it
    if projected_position.ndim == 0:
        transformed_position_delta = (projected_position_transformed - projected_position) * velocity_direction
    elif projected_position.ndim >= 1:
        transformed_position_delta = numpy.outer(projected_position_transformed - projected_position, velocity_direction)

    position_transformed = position + transformed_position_delta

    return position_transformed, time_transformed


