
using UnityEngine;

public static class GIUtility
{
    public static Vector4 ComputeUVToViewPos(Camera renderCamera)
    {
        float tanHalfFovY = Mathf.Tan(renderCamera.fieldOfView * Mathf.Deg2Rad * 0.5f);
        float tanHalfFovX = tanHalfFovY * renderCamera.aspect;

        return new Vector4(2 * tanHalfFovX, 2 * tanHalfFovY, -tanHalfFovX, -tanHalfFovY);
    }
}
