import textwrap

from handwriting_synthesis.hand import Hand

text = """Projectile motion is a form of motion experienced by an object or particle (a projectile) that is projected 
in a gravitational field, such as from Earth's surface, and moves along a curved path under the action of gravity 
only. In the particular case of projectile motion on Earth, most calculations assume the effects of air resistance 
are passive and negligible. The curved path of objects in projectile motion was shown by Galileo to be a parabola, 
but may also be a straight line in the special case when it is thrown directly upward or downward. The study of such 
motions is called ballistics, and such a trajectory is a ballistic trajectory. The only force of mathematical 
significance that is actively exerted on the object is gravity, which acts downward, thus imparting to the object a 
downward acceleration towards the Earth's center of mass. Because of the object's inertia, no external force is 
needed to maintain the horizontal velocity component of the object's motion. Taking other forces into account, 
such as aerodynamic drag or internal propulsion (such as in a rocket), requires additional analysis. A ballistic 
missile is a missile only guided during the relatively brief initial powered phase of flight, and whose remaining 
course is governed by the laws of classical mechanics."""

lines = textwrap.wrap(text, width=75)

if __name__ == '__main__':
    hand = Hand()

    # varying bias, varying style
    lines = lines
    biases = [.9 for i in lines]
    styles = [5 for i in lines]
    # biases = 0.0000000000003 * np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    # styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/text.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
